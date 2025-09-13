from lg_spectra.library import Library
from lg_spectra.matching import match_spectrum


def test_match_spectrum_ranking_and_color():
    lib = Library()
    ref = {
        'dct16': [1.0] + [0.0] * 15,
        'ratio': [1.0, 0.5, 0.2],
        'phash': 'phash_v1:0',
        'purity': 0.9,
        'rt': 10.0,
    }
    lib.add('good', ref)
    lib.add('bad', {
        'dct16': [0.0] * 16,
        'ratio': [0.0, 0.0, 0.0],
        'phash': 'phash_v1:ffffffffffffffff',
        'purity': 0.1,
        'rt': 20.0,
    })
    lib._finalise()
    res = match_spectrum(ref, lib)
    assert res[0]['id'] == 'good'
    assert res[0]['ampel'] == 'green'

