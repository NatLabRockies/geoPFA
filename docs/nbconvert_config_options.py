"""nbconvert config file"""

c = get_config()  # noqa: F821
c.NbConvertApp.notebooks = [
    "../examples/Newberry/3D/notebooks/5-newberry_superhot_400c.ipynb",
    "../examples/Nevada/2D/notebooks/5-nevada_conventional_150c_3km.ipynb",
    "../examples/Nevada/2D/notebooks/6-nevada_superhot_350c_7km.ipynb",
]
c.NbConvertApp.export_format = "rst"
c.NbConvertApp.recursive_glob = True
c.NbConvertApp.output_files_dir = "."
c.FilesWriter.build_directory = "./source/notebooks/"
