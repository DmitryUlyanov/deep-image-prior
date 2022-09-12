for i in {0..3}; do \
  /mnt5/nimrod/deep-image-prior/venv/bin/python3 denoising.py --gpu 4 --index $i --input_index 1; \
  /mnt5/nimrod/deep-image-prior/venv/bin/python3 Inpainting.py --gpu 4 --index $i; \
  /mnt5/nimrod/deep-image-prior/venv/bin/python3 SR.py --gpu 4 --index $i --input_index 1; \
done
