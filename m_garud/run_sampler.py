# import argparse
# from m_garud.data.sampler import sample_language_data


# def main():
#     parser = argparse.ArgumentParser("m-GARuD Dataset Sampler")
#     parser.add_argument(
#         "--lang",
#         type=str,
#         required=True,
#         help="Tribal language (e.g., bhili, gondi)"
#     )
#     parser.add_argument(
#         "--n",
#         type=int,
#         required=True,
#         help="Number of samples"
#     )
#     parser.add_argument(
#         "--tag",
#         type=str,
#         default=None,
#         help="Optional tag for output file"
#     )

#     args = parser.parse_args()

#     sample_language_data(
#         tribal_lang=args.lang.lower(),
#         n_samples=args.n,
#         output_suffix=args.tag
#     )


# if __name__ == "__main__":
#     main()


from m_garud.data.sampler import sample_language_data
# Assuming your previous config block is in a file named config.py
# or defined at the top of this script.

def run_sampling():
    # =========================
    # SAMPLING CONFIGURATION
    # =========================
    # You can change these values manually here
    TARGET_LANG = "bhili"  # e.g., "bhili" or "gondi"
    NUM_SAMPLES = 1000     # Or use DEFAULT_SAMPLE_SIZE from your config
    TAG = "initial_run"    # Optional suffix
    
    print(f"🚀 Starting sampling process for: {TARGET_LANG}")
    print(f"📊 Target samples: {NUM_SAMPLES}")
    print(f"🏷️  Tag: {TAG}")

    try:
        sample_language_data(
            tribal_lang=TARGET_LANG.lower(),
            n_samples=NUM_SAMPLES,
            output_suffix=TAG
        )
        print(f"✅ Sampling completed successfully for {TARGET_LANG}!")
        
    except Exception as e:
        print(f"❌ Error during sampling: {e}")

if __name__ == "__main__":
    run_sampling()