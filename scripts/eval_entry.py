from runtime import bootstrap, local_tokenizer, validate_real_cameras

bootstrap()
local_tokenizer()
validate_real_cameras()

if __name__ == "__main__":
    from lerobot.scripts.lerobot_eval import main
    main()
