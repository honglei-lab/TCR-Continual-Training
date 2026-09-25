from runtime import bootstrap, fixed_lora_seed, local_tokenizer, preserve_checkpoint_normalizers, validate_real_cameras

bootstrap()
fixed_lora_seed()
local_tokenizer()
preserve_checkpoint_normalizers()
validate_real_cameras()

if __name__ == "__main__":
    from lerobot.scripts.lerobot_train import main
    main()
