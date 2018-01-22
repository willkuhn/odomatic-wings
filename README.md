# odomatic-wings
Ongoing project for automatically recognizing species of dragonflies based on their wings

## About
This project is based on my dissertation work ([Dept. Biology](https://www.ncas.rutgers.edu/biology), Rutgers University-Newark, 2015), entitled *Three approaches to automating taxonomy, with emphasis on the Odonata (dragonflies & damselflies)*, and represents an ongoing project. It's a bit rough now, but check back later.

## Getting Started
### Required Packages
autoID was written for Python 3.X and may be mostly compatible with Python 2.7. Required packages are listed [here](autoID/requirements.txt).

### Installing
If you have [git](https://git-scm.com/), you can clone the git repository to a local directory, go to that directory, and install like this:
```sh
git clone https://github.com/willkuhn/odomatic-wings.git
cd odomatic-wings
python setup.py install
```

Alternatively, you can download the repository, extract it, navigate to the extracted folder in the terminal or command prompt, and install by typing:
```sh
python setup.py install
```

Now you can test that wingrid was sucessfully installed in Python:
```python
>>> import autoID
>>> autoID.__version__
'0.1.0'
```

## Author
Will Kuhn
willkuhn@crossveins.com
https://crossveins.com
