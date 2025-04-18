import wandb
import matplotlib.pyplot as plt
import re

# 로그인 (환경변수에 W&B API key가 설정돼 있다면 생략 가능)
wandb.login()

# 실험 정보 불러오기
api = wandb.Api()
run = api.run("opt_bs/LoRA_rank_test/65b5057n")  # 예: "effl/LoRA/abc123"

# 로그 데이터 가져오기
history = run.history(samples=1000)  # 필요한 만큼의 step 수 (너무 크면 오래 걸릴 수 있음)

# 특정 step만 필터링
target_step = 700
row = history[history["_step"] == target_step]

# ER 관련 컬럼만 추출
pattern = re.compile(r"ER/LoRAUpdate/module\.base_model\.model\.model\.layers\.(\d+)\.(\w+)_proj")

data = {}
for col in row.columns:
    match = pattern.match(col)
    if match:
        layer = int(match.group(1))
        proj = match.group(2)
        if proj not in data:
            data[proj] = {}
        data[proj][layer] = row[col].values[0]

# 시각화
for proj, layer_dict in data.items():
    layers = sorted(layer_dict.keys())
    values = [layer_dict[l] for l in layers]
    plt.plot(layers, values, label=proj)

plt.xlabel("Layer")
plt.ylabel("Effective Rank (ER)")
plt.title(f"Effective Rank per Layer at Step {target_step}")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig("./test.png")
