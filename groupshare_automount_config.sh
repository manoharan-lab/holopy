#!/bin/sh

#create script for mounting groupshare
cat > /home/$USER/mountgroupshare.sh <<EOF
#!/bin/sh
gvfs-mount smb://manoharanfs1.rc.fas.harvard.edu/manoharanfs1/
EOF

cat /home/$USER/mountgroupshare.sh

#make script executable
chmod +x /home/$USER/mountgroupshare.sh

#run this script to mount groupshare
sh /home/$USER/mountgroupshare.sh

#create a symbolic link to the groupshare
ln -s /run/user/$(id -u)/gvfs/smb-share:server=manoharanfs1.rc.fas.harvard.edu,share=manoharanfs1/ /home/$USER/group

#create an autostart entry for this script
cat > /home/$USER/.config/autostart/mountgroupshare.sh.desktop <<EOF
[Desktop Entry]
Type=Application
Exec=/home/$USER/mountgroupshare.sh
Hidden=false
NoDisplay=false
X-GNOME-Autostart-enabled=true
Name[en_US]=AutoMountGroupShare
Name=AutoMountGroupShare
Comment[en_US]=
Comment=
EOF

cat /home/$USER/.config/autostart/mountgroupshare.sh.desktop

echo 
echo 'DONE!! You should now be able to see the manoharanfs1 groupshare in Nautilus.' 
echo 'There is a also a link to the groupshare at ~/group'
echo 'To test the automounting, logout and log back in. If the groupshare is mounted, it worked.'
echo 'Contact Aaron (agoldfain@seas.harvard.edu) if there are any problems'

