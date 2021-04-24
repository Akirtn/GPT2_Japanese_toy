import gpt2_predictor
predictor = gpt2_predictor.Gpt2Predictor()
result = simple_generate_json.generate_json({"previous": "chocobo君は"})
print(result)