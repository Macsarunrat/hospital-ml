import subprocess

print("=" * 60)
print("Submitting kernel to Kaggle via CLI...")
print("=" * 60)

subprocess.run(["kaggle", "kernels", "push", "-p", "."], check=True)

print("\n" + "-" * 60)
print("Kernel successfully pushed to Kaggle.")
print("Check status:")
print("   kaggle kernels status macsarun/hospital-ml-training")
print("Download outputs after completion:")
print("   kaggle kernels output macsarun/hospital-ml-training -p rice/output/")
print("-" * 60)
