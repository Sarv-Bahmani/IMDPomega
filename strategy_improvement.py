from imdp import IMDP
from automata import Automata
from product import Product


address = 'Ab_UAV_10-16-2025_20-48-14'
noise_samples = 20000
I = IMDP(address=address, noise_samples=noise_samples)

all_labsets = {I.label[s] for s in I.states}
B = Automata(all_labsets, "my_automaton.hoa")
print('will build product')
P = Product(I, B)
print('product is build')
