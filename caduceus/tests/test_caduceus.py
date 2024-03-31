from transformers import AutoModelForMaskedLM, AutoTokenizer


if __name__ == "__main__":
    

    # See the `Caduceus` collection page on the hub for list of available models.
    model_name = "kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    print(model)