for i in {0..3}; do \
  /mnt5/nimrod/deep-image-prior/venv/bin/python3 denoising.py --gpu 0 --index $i --input_index 3; \
  /mnt5/nimrod/deep-image-prior/venv/bin/python3 Inpainting.py --gpu 0 --index $i; \
  /mnt5/nimrod/deep-image-prior/venv/bin/python3 SR.py --gpu 0 --index $i --input_index 3; \
done
