import os
import torch
from tqdm import tqdm
from signjoey.helpers import load_config, make_logger
from signjoey.sinhala_sentence.translation_model import SinhalaSignTranslationModel
from signjoey.sinhala_sentence.data import load_test_data
from signjoey.sinhala_sentence.search import greedy_decode
from signjoey.metrics import bleu, rouge
import random
import csv

def test_translation_model(cfg_file: str):
    cfg = load_config(cfg_file)
    train_config = cfg['training']
    
    # Load logger
    logger = make_logger(model_dir=train_config["model_dir"], log_file=f"{cfg['name']}_test.log")
    
    device = torch.device("cuda" if torch.cuda.is_available() and train_config.get("use_cuda", False) else "cpu")

    # Load test data
    test_loader, tokenizer = load_test_data(cfg, logger)
    
    # Load model
    model = SinhalaSignTranslationModel(cfg, logger)

    checkpoint_path = os.path.join(cfg["training"]["model_dir"], 'best_model.pth')
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model_state_dict'])
    logger.info(f"checkpoint loaded from {checkpoint_path}")
    model.to(device)
    model.eval()
    
    logger.info("Model loaded for testing.")
    
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"): 
            keypoints = batch['keypoints'].to(device)
            keypoints_mask = batch['keypoints_mask'].to(device)
            label = batch['label'].to(device)
            
            # Perform greedy decoding
            encoder_output = model.encode(keypoints, keypoints_mask)[0]
            decoded_sequences = greedy_decode(
                src_mask=keypoints_mask,
                bos_index=tokenizer.lang_code_to_id.get(cfg['data']['tgt_language']),
                eos_index=tokenizer.eos_token_id,
                max_output_length=cfg['data']['max_sent_length'],
                decoder=model.decoder,
                encoder_output=encoder_output,
                device=device
            )
            
            ref_text = tokenizer.batch_decode(label.squeeze(1).detach().cpu().numpy(), skip_special_tokens=True)
            hyp_text = tokenizer.batch_decode(decoded_sequences, skip_special_tokens=True)
            
            references.extend(ref_text)
            hypotheses.extend(hyp_text)
    
    # Compute BLEU and ROUGE scores
    bleu_scores = bleu(references, hypotheses)
    rouge_score = rouge(references, hypotheses)
    
    # Save references and hypotheses to CSV
    csv_path = os.path.join(cfg["training"]["model_dir"], 'test_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Reference', 'Hypothesis'])  # header
        for ref, hyp in zip(references, hypotheses):
            writer.writerow([ref, hyp])
    logger.info(f"Test results saved to {csv_path}")
    
    logger.info(f">> BLEU-1: {bleu_scores['bleu1']:.4f}, BLEU-2: {bleu_scores['bleu2']:.4f}, BLEU-3: {bleu_scores['bleu3']:.4f}, BLEU-4: {bleu_scores['bleu4']:.4f}, ROUGE: {rouge_score:.4f}")

    sample_indices = random.sample(range(len(references)), min(10, len(references)))
    samples = [(references[i], hypotheses[i]) for i in sample_indices]

    logger.info(">> Sample Sentences:")
    for sample in samples:
        logger.info(f"Reference: {sample[0]} - Hypothesis: {sample[1]}")


    logger.info("Testing completed.")

