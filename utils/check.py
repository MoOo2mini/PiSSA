from transformers import AutoModelForCausalLM
import torch

m1 = AutoModelForCausalLM.from_pretrained("output/metamath-PiSSA-Llama-2-7b-r128_1280_hi", trust_remote_code=True)
m2 = AutoModelForCausalLM.from_pretrained("output/metamath-PiSSA-Llama-2-7b-r128_1280_hi_merged10", trust_remote_code=True)

sd1 = m1.state_dict()
sd2 = m2.state_dict()

for k in sd1:
    if not torch.allclose(sd1[k], sd2[k], atol=1e-6):
        print(sd1[k], sd2[k])
        print("🔍 Difference found in:", k)
        break
print("SAME!")
# from transformers import AutoTokenizer

# t1 = AutoTokenizer.from_pretrained("output/metamath-PiSSA-Llama-2-7b-r128_1280")
# t2 = AutoTokenizer.from_pretrained("output/metamath-PiSSA-Llama-2-7b-r128_1280_temp_fixed")

# print(t1.special_tokens_map == t2.special_tokens_map)
# print(t1.all_special_tokens == t2.all_special_tokens)
