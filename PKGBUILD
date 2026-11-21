# Maintainer: Mike Falls <96034740+drowsyclimber@users.noreply.github.com>
pkgname=mathwave
pkgver=0.1.0
pkgrel=4
pkgdesc="Cava-style terminal audio visualizer that drives a grid of math functions and prints their formulas"
arch=('any')
url="https://github.com/drowsyclimber/mathwave"
license=('MIT')
depends=('python' 'python-numpy')
makedepends=('python-build' 'python-installer' 'python-setuptools' 'python-wheel')
checkdepends=('python-pytest')
optdepends=(
  'libpulse: parec capture (works with PulseAudio and pipewire-pulse)'
  'pipewire: pw-record capture'
  'alsa-utils: aplay output for --demo --audible'
)
# Local build: run `makepkg -si` next to mathwave-0.1.0.tar.gz.
# For the AUR, swap this for the GitHub release tarball (see packaging/PKGBUILD.aur).
source=("$pkgname-$pkgver.tar.gz")
sha256sums=('SKIP')

build() {
  cd "$pkgname-$pkgver"
  python -m build --wheel --no-isolation
}

check() {
  cd "$pkgname-$pkgver"
  PYTHONPATH="$PWD/src" python -m pytest -q
}

package() {
  cd "$pkgname-$pkgver"
  python -m installer --destdir="$pkgdir" dist/*.whl
  install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
  install -Dm644 man/mathwave.1 "$pkgdir/usr/share/man/man1/mathwave.1"
  install -Dm644 config/config "$pkgdir/usr/share/doc/$pkgname/config.example"
  install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"
}
