"""Portable command-line training/evaluation shared by dataset entry points."""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.nn import functional as F
from .objectives import DistillationObjective, METHODS, DEFAULT_WEIGHTS
from ._indistill_core import widen_student

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {
    "resnet56_to_resnet20": ("resnet56", "resnet20"),
    "resnet110_to_resnet32": ("resnet110", "resnet32"),
    "resnet32x4_to_resnet8x4": ("resnet32x4", "resnet8x4"),
    "wrn_40_2_to_wrn_16_2": ("wrn_40_2", "wrn_16_2"),
    "wrn_40_2_to_wrn_40_1": ("wrn_40_2", "wrn_40_1"),
    "vgg13_to_vgg8": ("vgg13", "vgg8"),
    "resnet32x4_to_ShuffleV1": ("resnet32x4", "ShuffleV1"),
    "wrn_40_2_to_ShuffleV1": ("wrn_40_2", "ShuffleV1"),
    "vgg13_to_mobilenetv1": ("vgg13", "mobilenetv1"),
    "ResNet50_to_mobilenetv1": ("ResNet50", "mobilenetv1"),
    "resnet32x4_to_ShuffleV2": ("resnet32x4", "ShuffleV2"),
}


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save(path, payload, binary=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if binary:
        torch.save(payload, temporary)
    else:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(10):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1 * (attempt + 1))


def checkpoint(path):
    # Only load checkpoints supplied by the experiment owner.
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model(model, path):
    state = checkpoint(path)
    for key in ("model", "state_dict", "student", "net"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def guard(path, settings):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != settings:
            raise RuntimeError(f"Configuration/source changed: {path}. Select a new --output-root.")
    else:
        save(path, settings)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    count, correct1, correct5, loss = 0, 0, 0, 0.
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        guesses = logits.topk(5, 1).indices
        count += len(labels)
        correct1 += (guesses[:, 0] == labels).sum().item()
        correct5 += (guesses == labels[:, None]).any(1).sum().item()
        loss += F.cross_entropy(logits.float(), labels, reduction="sum").item()
    if not count:
        raise ValueError("Evaluation dataset is empty")
    return dict(top1=100 * correct1 / count, top5=100 * correct5 / count, loss=loss / count, samples=count)


def train_one(args, student, teacher, method, alter, train_loader, val_loader, test_loader,
              data_module, output, protocol, *, ce=None, distill=None):
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(protocol=protocol, method=method, alter=alter, ce=ce, distill=distill)
    guard(output / "protocol.json", settings)
    complete = output / "result.json"
    if complete.exists():
        print(f"[SKIP] {output}", flush=True)
        return
    seed_all(args.seed)
    student.to(args.device)
    if teacher is not None:
        teacher.to(args.device).requires_grad_(False).eval()
    objective = DistillationObjective(student, teacher, method, alter=alter,
        student_name=args.student, teacher_name=args.teacher, input_size=args.input_size,
        ce=ce, distill=distill, response=args.response_weight).to(args.device)
    if method == "InDistill" and args.epochs <= objective.extra["information"].warmup_epochs:
        raise ValueError("The run must continue beyond the InDistill intermediate warmup")
    optimizer = torch.optim.SGD(objective.learnable_parameters(), lr=args.lr, momentum=.9, weight_decay=.0005)
    start, history, seconds, peak_memory = 0, [], 0., 0
    last = output / "last.pt"
    if last.exists():
        state = checkpoint(last)
        if state["settings"] != settings:
            raise RuntimeError("Resume settings differ")
        student.load_state_dict(state["model"])
        objective.load_auxiliary_state(state["auxiliary"])
        optimizer.load_state_dict(state["optimizer"])
        start, history, seconds = state["epoch"], state["history"], state["seconds"]
        peak_memory = state.get("peak_memory", 0)
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.set_rng_state(state["rng"]["torch"])
        if torch.cuda.is_available() and state["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng"]["cuda"])
    save(output / "feature_layout.json", objective.layout)
    for epoch in range(start, args.epochs):
        objective.train()
        data_module.prepare_epoch(train_loader, args.seed, epoch)
        lr = args.lr * .1 ** sum(epoch >= stage for stage in args.milestones)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        totals, count = {}, 0
        for step, (images, target) in enumerate(train_loader, 1):
            images, target = images.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            _, losses, metrics = objective(images, target, epoch)
            loss = sum(losses.values())
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {step}; last completed epoch is resumable")
            loss.backward()
            for parameter in objective.learnable_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                    raise FloatingPointError("Non-finite gradient; the incomplete epoch will restart on resume")
            optimizer.step()
            count += len(target)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.) + value.item() * len(target)
            if step == 1 or step % args.log_every == 0:
                print(f"[{method}{'+ALTER' if alter else ''}] epoch {epoch + 1}/{args.epochs} "
                      f"batch {step}/{len(train_loader)} loss={loss.item():.5f}", flush=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
        elapsed = time.perf_counter() - began
        seconds += elapsed
        row = dict(epoch=epoch + 1, lr=lr, train_seconds=elapsed,
                   losses={key: value / count for key, value in totals.items()}, metrics=metrics)
        if val_loader is not None:
            row["validation"] = evaluate(student, val_loader, args.device)
        history.append(row)
        payload = dict(model=cpu_state(student), auxiliary=objective.auxiliary_state(), optimizer=optimizer.state_dict(),
                       epoch=epoch + 1, settings=settings, history=history, seconds=seconds, peak_memory=peak_memory,
                       rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
        save(last, payload, binary=True)
        save(output / "history.json", history)
    # Only final checkpoint selection. No best-test or best-validation selection.
    save(output / "student_final.pth", dict(model=cpu_state(student), epoch=args.epochs, protocol=settings), binary=True)
    result = dict(method=method, alter=alter, seed=args.seed, epoch=args.epochs, selection="final",
                  training_seconds=seconds, peak_memory_mib=peak_memory / 2**20,
                  test=evaluate(student, test_loader, args.device), protocol_sha256=digest(settings))
    save(complete, result)
    print(f"[DONE] {method}{'+ALTER' if alter else ''}: {result['test']['top1']:.2f}%", flush=True)


def parser(dataset, mode):
    p = argparse.ArgumentParser(description=f"ALTER {dataset}: {mode}")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=ROOT / ("CIFAR-100" if dataset == "cifar" else "ImageNet") / "Utils/runs")
    p.add_argument("--pair", choices=tuple(PAIRS))
    p.add_argument("--all-pairs", action="store_true")
    p.add_argument("--student", default="resnet8x4" if dataset == "cifar" else "resnet18")
    p.add_argument("--teacher", default="resnet32x4" if dataset == "cifar" else "resnet34")
    p.add_argument("--model", help="Model name for supervised training or testing")
    p.add_argument("--method", choices=METHODS, default="KD")
    p.add_argument("--methods", choices=METHODS, nargs="+")
    p.add_argument("--alter", action="store_true")
    p.add_argument("--compare", action="store_true", help="Run matched baseline and +ALTER from fresh identical initialization")
    p.add_argument("--teacher-checkpoint", type=Path)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--checkpoint", type=Path, help="Student checkpoint for evaluation")
    p.add_argument("--epochs", type=int, default=240 if dataset == "cifar" else 100)
    p.add_argument("--batch-size", type=int, default=128 if dataset == "cifar" else 256)
    p.add_argument("--lr", type=float, default=.01 if dataset == "cifar" else .1)
    p.add_argument("--milestones", type=int, nargs="+", default=[150, 180, 210] if dataset == "cifar" else [30, 60, 90])
    p.add_argument("--response-weight", type=float, default=200.)
    p.add_argument("--ce-weight", type=float)
    p.add_argument("--distill-weight", type=float)
    p.add_argument("--base-augmentation", choices=("concat", "weak"), default="concat")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--download", action="store_true", help="Explicitly allow CIFAR-100 download")
    return p


def main(dataset="cifar", mode="distillation"):
    args = parser(dataset, mode).parse_args()
    args.smoke, args.val_per_class, args.split_seed = False, 50, 2021
    args.input_size = 32 if dataset == "cifar" else 224
    if args.epochs < 1 or args.batch_size < 2 or args.workers < 0 or args.log_every < 1:
        raise ValueError("Invalid epochs, batch size, workers or logging interval")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate a compatible PyTorch environment")
    if args.alter and args.compare:
        raise ValueError("Use --alter or --compare, not both")
    if mode == "supervised" and (args.alter or args.compare):
        raise ValueError("ALTER must be attached to a distillation objective")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    folder = ROOT / ("CIFAR-100" if dataset == "cifar" else "ImageNet")
    sys.path.insert(0, str(folder / "Utils"))
    models = load_module("_alter_" + dataset + "_models", folder / "Models/__init__.py", package=True)
    # A normally importable module name is required when DataLoader workers
    # pickle dataset instances under Windows spawn. Utils is already on sys.path.
    data = importlib.import_module("data")
    if args.download and dataset == "cifar":
        from torchvision.datasets import CIFAR100
        CIFAR100(str(args.data_root), train=True, download=True)
        CIFAR100(str(args.data_root), train=False, download=True)
    train_loader, val_loader, test_loader, split = data.make_data(args, args.seed)
    if mode == "test":
        if args.checkpoint is None:
            raise ValueError("Pass --checkpoint")
        model = models.build_model(args.model or args.student).to(args.device)
        load_model(model, args.checkpoint)
        print(json.dumps(evaluate(model, test_loader, args.device), indent=2))
        return
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): sha_file(path)
               for directory in (ROOT / "Distiller", folder / "Models", folder / "Utils")
               for path in directory.glob("*.py")}
    if mode == "supervised":
        pair_list = [(args.model or args.student, None, args.model or args.student)]
    elif args.all_pairs:
        if dataset != "cifar":
            raise ValueError("--all-pairs is the CIFAR-100 11-pair matrix")
        pair_list = [(key, teacher, student) for key, (teacher, student) in PAIRS.items()]
    elif dataset == "cifar":
        selected_pair = args.pair or (args.teacher + "_to_" + args.student)
        if selected_pair not in PAIRS:
            raise ValueError("Select one of the documented eleven CIFAR pairs with --pair")
        teacher, student = PAIRS[selected_pair]
        pair_list = [(selected_pair, teacher, student)]
    else:
        pair_list = [(args.teacher + "_to_" + args.student, args.teacher, args.student)]
    for pair, teacher_name, student_name in pair_list:
        args.teacher, args.student = teacher_name, student_name
        methods = ["CE"] if mode == "supervised" else (args.methods or [args.method])
        teacher = None
        teacher_hash = None
        if teacher_name is not None:
            path = args.teacher_checkpoint if not args.all_pairs else None
            if path is None:
                options = [args.checkpoint_dir / name for name in
                           (teacher_name + "_ckpt_epoch_240.pth", teacher_name + ".pth", teacher_name + ".pth.tar")]
                path = next((candidate for candidate in options if candidate.is_file()), None)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"Supply {teacher_name} via --teacher-checkpoint or --checkpoint-dir")
            teacher = models.build_model(teacher_name)
            load_model(teacher, path)
            teacher_hash = sha_file(path)
        protocol = dict(dataset=dataset, teacher=teacher_name, student=student_name, teacher_sha256=teacher_hash,
                        seed=args.seed, split=split, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                        milestones=args.milestones, augmentation=args.base_augmentation, response_weight=args.response_weight,
                        precision="FP32", source_sha256=digest(sources), torch_version=str(torch.__version__))
        for method in methods:
            if dataset == "imagenet" and method == "InDistill":
                raise ValueError("ImageNet InDistill uses a separate historical implementation and is outside this release")
            seed_all(args.seed)
            prototype = models.build_model(student_name)
            initial = cpu_state(prototype)
            active_teacher, active_teacher_name = teacher, teacher_name
            # Shared 2x-width auxiliary teacher for the five heterogeneous pairs.
            if method == "InDistill" and PAIRS.get(pair) and list(PAIRS).index(pair) >= 6:
                aux = widen_student(prototype)
                auxiliary_root = args.output_root / pair / "InDistill_auxiliary"
                train_one(args, aux, copy.deepcopy(teacher), "KD", False, train_loader, val_loader, test_loader,
                          data, auxiliary_root, protocol, ce=1., distill=16.)
                load_model(aux, auxiliary_root / "student_final.pth")
                aux.cpu()
                active_teacher, active_teacher_name = aux, student_name
                protocol_aux = {**protocol, "auxiliary_sha256": sha_file(auxiliary_root / "student_final.pth")}
            else:
                protocol_aux = protocol
            for alter in ([False, True] if args.compare else [args.alter]):
                seed_all(args.seed)
                student = models.build_model(student_name)
                student.load_state_dict(initial)
                args.teacher = active_teacher_name
                ce, distill = args.ce_weight, args.distill_weight
                if method == "MLKD" and distill is None and student_name in ("mobilenetv1", "ShuffleV2"):
                    distill = .5  # recorded stability preset for these CIFAR pairs
                run_folder = args.output_root / pair / (method + ("_ALTER" if alter else "_baseline"))
                train_one(args, student, copy.deepcopy(active_teacher), method, alter, train_loader, val_loader,
                          test_loader, data, run_folder, protocol_aux, ce=ce, distill=distill)
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            args.teacher = teacher_name
