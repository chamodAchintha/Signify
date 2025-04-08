import os
import shutil
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from signjoey.helpers import load_config, make_logger, set_seed
from signjoey.sinhala_sentence.translation_model import SinhalaSignTranslationModel
from signjoey.builders import build_optimizer
from signjoey.early_stopping import EarlyStopping
from signjoey.sinhala_sentence.data import load_training_data
from signjoey.sinhala_sentence.search import greedy_decode
from signjoey.metrics import bleu, rouge

import random

def train_translation_model(cfg_file: str):

    cfg = load_config(cfg_file)
    train_config = cfg['training']

    # output path
    os.makedirs(train_config["model_dir"], exist_ok=True)

    shutil.copy2(cfg_file, train_config["model_dir"] + "/config.yaml")

    logger = make_logger(model_dir=train_config["model_dir"], log_file=f"{cfg['name']}_train.log")
    validation_file = f"{train_config['model_dir']}/{cfg['name']}_validation.txt"
    with open(validation_file, "w", encoding="utf-8") as opened_file:
        pass

    # set the random seed
    set_seed(seed=cfg["training"].get("random_seed", 42))

    train_loader, val_loader, tokenizer = load_training_data(cfg, logger)

    model = SinhalaSignTranslationModel(cfg, logger)
    logger.info(f'translation model created:\n{model}')

    use_cuda = cfg["training"].get("use_cuda", False)
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    logger.info(f"device: {device}")

    model.to(device)


    # optimization
    current_lr = train_config["learning_rate"]
    optimizer = build_optimizer(
        config=train_config,
        parameters=filter(lambda p: p.requires_grad, model.parameters())
    )

    criterion = torch.nn.CrossEntropyLoss()

    # learning rate scheduling
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        verbose=False,
        threshold_mode="abs",
        factor=train_config.get("decrease_factor", 0.1),
        patience=train_config.get("patience", 5),
        min_lr=train_config.get("learning_rate_min", 1e-6)
    )

    num_epochs = train_config["epochs"]
    best_val_loss = float('inf')

    logger.info(f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info("Training Starts...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        current_lr = optimizer.param_groups[0]['lr']

        for batch in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            keypoints = batch['keypoints'].to(device)
            keypoints_mask = batch['keypoints_mask'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            logits, _ = model(
                sgn = keypoints,
                sgn_mask = keypoints_mask,
                text_input_ids = text_input_ids,
                text_attention_mask = text_attention_mask,
                label = label
            )
            loss = None
            loss = criterion(logits.view(-1, model.decoder_config.vocab_size), label.view(-1))
            loss.backward() 

            optimizer.step() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # validation
        tgt_lang_code = cfg['data']['tgt_language']
        max_output_length = cfg['data']['max_sent_length']
        avg_val_loss, bleu_scores, rouge_score, samples = validate_model(
            model=model, 
            val_loader=val_loader, 
            criterion=criterion, 
            tokenizer=tokenizer, 
            tgt_lang_code=tgt_lang_code, 
            device=device, 
            max_output_length=max_output_length
        )

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}, lr: {current_lr:.6f}')
        logger.info(f">> BLEU-1: {bleu_scores['bleu1']:.4f} BLEU-2: {bleu_scores['bleu2']:.4f} BLEU-3: {bleu_scores['bleu3']:.4f} BLEU-4: {bleu_scores['bleu4']:.4f} ROUGE: {rouge_score:.4f}")
        logger.info(">> Sample Sentences:")
        for sample in samples:
            logger.info(f"Reference: {sample[0]} - Hypothesis: {sample[1]}")

        with open(validation_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f} Validation Loss: {avg_val_loss:.4f}, lr: {current_lr:.6f}\n')

        # save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'learning_rate': optimizer.param_groups[0]['lr'],
            }
            torch.save(checkpoint, os.path.join(train_config["model_dir"], 'best_model.pth'))
            logger.info(f"New best model saved with validation loss {best_val_loss:.4f}")

        # Step the scheduler
        scheduler.step(avg_val_loss)



        # if device.type == 'cuda':
        #     torch.cuda.empty_cache()
    logger.info('Training Completed.')

def validate_model(model, val_loader, criterion, tokenizer, tgt_lang_code='si_LK', device = 'cpu', max_output_length=30, sample_count=5):
    model.eval()
    total_val_loss = 0
    hypotheses = []
    references = []

    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader), desc=f"Validation: "):
            keypoints = batch['keypoints'].to(device)
            keypoints_mask = batch['keypoints_mask'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            label = batch['label'].to(device)

            logits, _ = model(
                sgn = keypoints,
                sgn_mask = keypoints_mask,
                text_input_ids = text_input_ids,
                text_attention_mask = text_attention_mask,
                label = label
            )

            val_loss = criterion(logits.view(-1, model.decoder_config.vocab_size), label.view(-1))
            total_val_loss += val_loss.item()

            # Perform greedy decoding
            encoder_output = model.encode(keypoints, keypoints_mask)[0]
            decoded_sequences = greedy_decode(
                src_mask=keypoints_mask,
                bos_index=tokenizer.lang_code_to_id.get(tgt_lang_code),
                eos_index=tokenizer.eos_token_id,
                max_output_length=max_output_length,
                decoder=model.decoder,
                encoder_output=encoder_output,
                device=device
            )
            
            ref_text = tokenizer.batch_decode(label.squeeze(1).detach().cpu().numpy(), skip_special_tokens=True)
            hyp_text = tokenizer.batch_decode(decoded_sequences, skip_special_tokens=True)
            references.extend(ref_text)
            hypotheses.extend(hyp_text)
        
    avg_val_loss = total_val_loss / len(val_loader)
    bleu_scores = bleu(references, hypotheses)
    rouge_score = rouge(references, hypotheses)

    # Select random samples
    sample_indices = random.sample(range(len(references)), min(sample_count, len(references)))
    samples = [(references[i], hypotheses[i]) for i in sample_indices]

    return avg_val_loss, bleu_scores, rouge_score, samples
