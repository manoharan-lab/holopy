from subprocess import Popen, PIPE
import os.path

major_version = "1.9"

holopy_dir = os.path.split(__file__)[0]

try:
    # attempt to get a minor version number based on the bzr revision number
    revno = Popen(['bzr','revno'],stdout=PIPE,
                  cwd=holopy_dir).communicate()[0].strip()
except OSError:
    # couldn't call bzr, so just set the revno to 0
    revno = 0
if revno == '':
    revno = 0
    
__version__ = major_version+"."+str(revno)
__version_info__ = tuple([ int(num) for num in __version__.split('.')])

