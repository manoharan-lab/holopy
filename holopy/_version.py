from subprocess import Popen, PIPE

revno = Popen(['bzr','revno'],stdout=PIPE).communicate()[0].strip()
__version__ = "1.9."+str(revno)
__version_info__ = tuple([ int(num) for num in __version__.split('.')])
