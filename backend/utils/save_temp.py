import base64, tempfile


def save_base64_png(data_uri):
"""Save a data:image/png;base64,... string to a temp PNG file and return its path."""
header, b64 = data_uri.split(',', 1)
data = base64.b64decode(b64)
tf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
tf.write(data)
tf.flush(); tf.close()
return tf.name