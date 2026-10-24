# BME_TDK_2026_weak_qubit_readout
"A qubit kiolvasás mérési bizonytalanságának egy egyszerű modellje" című, a BME TTK 2026. évi TDK-jára leadott dolgozat kódbázisa

A fájlok egy részének beszédes neve van, úgyhogy csak a kevésbé egyértelműeket írom le. Az upscaled.py az eredeti klasszikus Monte Carlo, az avg_tree a klasszikus átlagolós inferenciára utal.
Az FN kezdetűek a F(N) függvényt számolják ki, azaz a fidelity függését az idősor hosszától. Amelyik nincs jelölve az a klasszikus, a kvantumosnak benne van a nevében. Az FNK ugyanez, csak implementálja az iterációt a szaturáció megtalálásához.
Hasonlóan az FQ F(q) függvényre, tehát a fidelity paraméterfüggésére utal. A kvantumos paraméterfüggés is örökli ezt a nevet, hiába nincsen benne q.

A simulate_quantum.py,  a qtree.py és final2.py működő, de optimalizálatlan kódok. Ezeket AI nélkül írtam, így nagyon lassan futnak, de talán emberi olvasók számára érthetőbbek.
