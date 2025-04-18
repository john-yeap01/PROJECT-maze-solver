from setuptools import setup

package_name = 'auto_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nus',
    maintainer_email='nus@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'r2mover = auto_nav.r2mover:main',
            'r2moverotate = auto_nav.r2moverotate:main',
            'r2scanner = auto_nav.r2scanner:main',
            'r2occupancy = auto_nav.r2occupancy:main',
            'r2occupancy2 = auto_nav.r2occupancy2:main',
            'r2auto_nav = auto_nav.r2auto_nav:main',
            'scanner = auto_nav.scanner:main',
	        'move = auto_nav.move:main',
	        'map2base = auto_nav.map2base:main',
	        'diyauto = auto_nav.auto_nav:main',
	        'occupancy = auto_nav.occupancy:main',
	        'solver = auto_nav.maze_solver:main',
	        'occupancy2 = auto_nav.occupancy2:main',
            'solver2 = auto_nav.solver2:main',
	        'find_bucket = auto_nav.find_bucket:main',
        ],
    },
)

