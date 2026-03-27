from setuptools import find_packages, setup
from glob import glob

package_name = 'autorace_idminer'

setup(
    name=package_name,
    version='26.3.24',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/param', glob('param/*.yaml')),
        ('share/' + package_name + '/map', glob('map/*.*')),
        ('share/' + package_name + '/image', glob('image/*.png')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='autorace tutorial code',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'detect_traffic_light = autorace_idminer.detect.detect_traffic_light:main',
            'detect_intersection = autorace_idminer.detect.detect_intersection:main',
            'detect_construction = autorace_idminer.detect.detect_construction:main',
            'detect_parking = autorace_idminer.detect.detect_parking:main',
            'detect_level = autorace_idminer.detect.detect_level:main',
            'detect_tunnel = autorace_idminer.detect.detect_tunnel:main',
            'detect_lane = autorace_idminer.detect.detect_lane:main',
            'control_lane = autorace_idminer.control.control_lane:main',
        ],
    },
)
