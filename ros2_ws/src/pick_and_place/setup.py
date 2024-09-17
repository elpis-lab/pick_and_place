from setuptools import find_packages, setup

package_name = 'pick_and_place'

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
    maintainer='sultan',
    maintainer_email='msultan@wpi.edu',
    description='Package for publishing end effector pose for pick and place task.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_sub = pick_and_place.depth_subscriber:main',
            'depth = pick_and_place.depth_map_pub:main',
        ],
    },
)
