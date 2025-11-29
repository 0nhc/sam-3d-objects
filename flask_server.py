import os
import sys
import uuid
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import io
import traceback

class SAM3DFlaskServer:
    def __init__(self):
        # Directory where we store per-request temp files
        self.work_root = "tmp_uploads"
        os.makedirs(self.work_root, exist_ok=True)

        # Load SAM 3D checkpoint (your Inference pipeline)
        self._load_checkpoint()

        # Flask related
        self.app = Flask(__name__)
        self.setup_routes()

    def _load_checkpoint(self):
        """
        Load your original demo pipeline and keep the objects on self so
        we don't re-import/reload the model for every request.
        """
        sys.path.append("notebook")
        from inference import Inference, load_image, load_single_mask

        tag = "hf"
        config_path = f"checkpoints/{tag}/pipeline.yaml"
        # Store these on self for later use
        self.inference = Inference(config_path, compile=False)
        self.load_image = load_image
        self.load_single_mask = load_single_mask

    def setup_routes(self):
        @self.app.route('/get_3d_mesh', methods=['POST'])
        def get_3d_mesh():
            """
            Expects multipart/form-data with:
            - files['image'] : RGB (or RGBA) image file
            - files['mask']  : mask image file
            - form['seed']   : (optional) integer seed
            Returns:
            - .ply file as attachment
            """
            try:
                if 'image' not in request.files or 'mask' not in request.files:
                    return jsonify({'error': 'Both "image" and "mask" files are required'}), 400

                image_file = request.files['image']
                mask_file = request.files['mask']

                if image_file.filename == '' or mask_file.filename == '':
                    return jsonify({'error': 'Empty filename for image or mask'}), 400

                # Optional seed parameter
                seed_str = request.form.get('seed', '42')
                try:
                    seed = int(seed_str)
                except ValueError:
                    return jsonify({'error': f'Invalid seed value: {seed_str}'}), 400

                # Create a per-request working directory
                request_id = uuid.uuid4().hex
                work_dir = os.path.join(self.work_root, request_id)
                os.makedirs(work_dir, exist_ok=True)

                image_dir = os.path.join(work_dir, "image")
                mask_dir = os.path.join(work_dir, "mask")
                os.makedirs(image_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)

                # --------- save image file normally ----------
                image_filename = secure_filename(image_file.filename) or "image.png"
                image_path = os.path.join(image_dir, image_filename)
                image_file.save(image_path)

                # Use the same loading utilities as in your working demo
                image = self.load_image(image_path)

                # --------- force mask to be saved as 0.png ----------
                # Read uploaded mask into memory and re-save as mask_dir/0.png
                mask_bytes = mask_file.read()
                if not mask_bytes:
                    return jsonify({'error': 'Uploaded mask file is empty'}), 400

                from PIL import Image
                import io

                mask_img = Image.open(io.BytesIO(mask_bytes))
                # If you know your masks should be single-channel, you can do:
                # mask_img = mask_img.convert("L")

                mask_index = 0
                mask_filename = f"{mask_index}.png"
                mask_path = os.path.join(mask_dir, mask_filename)
                mask_img.save(mask_path)

                # Now load_single_mask will find mask_dir/0.png
                mask = self.load_single_mask(mask_dir, index=mask_index)

                # Run the model
                output = self.inference(image, mask, seed=seed)

                # Export gaussian splat to .ply
                ply_path = os.path.join(work_dir, "splat.ply")
                output["gs"].save_ply(ply_path)

                # Return the .ply file directly to the client
                return send_file(
                    ply_path,
                    mimetype='application/octet-stream',
                    as_attachment=True,
                    download_name='splat.ply'
                )

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500


    def run(self):
        # For remote access, host='0.0.0.0'
        # Debug=True is fine for development, but turn it off for production
        self.app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    server = SAM3DFlaskServer()
    server.run()
