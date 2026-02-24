# twistjm
TWIST JM is a minimal neural operator based on additive integer algebra of J Operator ( Thorn, A. M. (2026). The Hyperbolic Automorphism J: A Golden Ratio Unit in ℤ[ζ₅]. Zenodo. https://doi.org/10.5281/zenodo.18523754 ).

## The First Run
Raw output from a 64-layer TWIST J model trained in 15 minutes on PC on the TinyStories dataset, running fully in INT8 integer algebra:

```
Filename: twist64epoch1.pt
[OK] Loaded (Steps: 4689)
==================================================
🎛️ TWIST-LLM LABORATORY V34.0 (THE INT8 CORE)
==================================================
Dataset : TinyStories
Model   : Dim=512 | Depth=64 | Seq=256 | Batch=32 | Shift=1
Status  : Trained (4689 steps)
--------------------------------------------------
[1] 📚 Change Dataset | [2] ⚙️ Edit Config | [3] 🧠 Train TWIST | [4] 💬 Interactive Chat
[5] 🔬 Quick Eval    | [6] 💾 Checkpoints | [8] ⏱️ RUN BENCHMARK | [9] 🛠️ TEST INT8 ENGINE
[7] 🚪 Exit

Select action: 4

[INFO] Engine: INT8 SIMD | Max Length: 400 tokens

Prompt: Once upon a time, a clever man built a tiny, fast robot. He did not use heavy blocks, only small pieces. He turned it on, and the little robot said,
TWIST: Once upon a time, a clever man built a tiny, fast robot. He did not use heavy blocks, only small pieces. He turned it on, and the little robot said,  "Your ugly pile is bad to be your best."
The robot thanked the man with the man and went to find a way to play with the robot. The robot was so old robot and the robot said, "We must fix your robot, it is so it." The robot was amazed and said, "Wow, that's not nice to be the robot. The robot was very happy too. It is a good job and it was a good job."
The robot thanked the robot for the robot and went to the robot. They played with the robot every day and the robot would be obedient.
<|endoftext|>

[Model ukončil text]
```

