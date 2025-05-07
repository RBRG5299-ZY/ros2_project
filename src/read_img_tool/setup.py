from setuptools import find_packages, setup

package_name = 'read_img_tool'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rbrg5299',
    maintainer_email='20232376@stu.neu.edu.cn',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'binocular_pub = read_img_tool.binocular:main',
            'monocular_pub = read_img_tool.monocular:main',
            'rgbd_pub = read_img_tool.rgbd:main'
        ],
    },
)
