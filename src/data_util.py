import json
from pathlib import Path
import string
from typing import Dict, Optional, List


def convert_jsonlist(src_path: Path, dst_path: Path):
    COMMENTS: str = "comments"
    BODY: str = "body"

    with open(src_path, 'r') as src, open(dst_path, 'w') as dst:
        for line in src:
            post = json.loads(line)
            if COMMENTS in post:
                for comment in post[COMMENTS]:
                    if BODY in comment:
                        dst.writelines([comment[BODY].lower()])


class SymbolIndexer:
    _known_symbol_to_index: Dict[str, int]
    _index_to_known_symbol: Dict[int, str]

    _unknown_idx: int
    _size: int

    def _add_symbol(self, symbol: str):
        self._known_symbol_to_index[symbol] = self._size
        self._index_to_known_symbol[self._size] = symbol
        self._size += 1

    def _add_unknown(self):
        self._unknown_idx = self._size
        self._size += 1

    def __init__(self, data: List[str]):
        self._size = 0
        self._known_symbol_to_index = {}
        self._index_to_known_symbol = {}

        self._add_unknown()
        for elem in data:
            self._add_symbol(elem)

    def size(self) -> int:
        return self._size

    def to_index(self, symbol: str) -> int:
        return self._known_symbol_to_index[symbol] if symbol in self._known_symbol_to_index else self._unknown_idx

    def to_symbol(self, index: int) -> Optional[str]:
        return self._index_to_known_symbol[index] if index in self._index_to_known_symbol else None

    @classmethod
    def english(cls):
        return cls([s for s in (string.ascii_letters + "0123456789" + ",.!? ")])

    @classmethod
    def spanish(cls):
        return cls([s for s in (string.ascii_letters + "0123456789" + "áéíóúñÁÉÍÓÚÑ¡¿üÜ" + ",.!? ")])

    @classmethod
    def russian(cls):
        return cls([s for s in ("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯаэыуояеёюибвгджзклмнпрстфхцчшщйьъ" + "0123456789" + ",.!? ")])
    
    @classmethod
    def japanese(cls):
        return cls([s for s in ("ーぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをんァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ゛゜" + "0123456789" + ",.!?")])

    @classmethod
    def chinese(cls):
        return cls([s for s in ("的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出而要于就下得可你年生自会那后能对着事其里所去行过家十用发天如然作方成者多日都三小军二无同么经法当起与好看学进种将还分此心前面又定见只主没公从" + "一二三四五六七八九十" + ",.!? ")])

    @classmethod
    def french(cls):
        return cls([s for s in (string.ascii_letters + "ùûüÿàâæçéèêëïîôœÚÛÜŸÁÂÆÇÉÈÊËÏÎÔŒ" + "0123456789" + ",.!? ")])
    
    @classmethod
    def norwegian(cls):
        return cls([s for s in ("'ÅÆØåæøABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'" + "0123456789" + ",.!? ")])

    @classmethod
    def hindi(cls):
        return cls([s for s in ("आइईउऊऋएऐओऔअंअःकककाकिकीकुकूकृकेकैकोकौकंकःखखखाखिखीखुखूखृखेखैखोखौखंखःगगगागिगीगुगूगृगेगैगोगौगंगःघघघाघिघीघुघूघृघेघैघोघौघंघःङङङाङिङीङुङूङृङेङैङोङौङंङःचचचाचिचीचुचूचृचेचैचोचौचंचःछछछाछिछीछुछूछृछेछैछोछौछंछःजजजाजिजीजुजूजृजेजैजोजौजंजःझझझाझिझीझुझूझृझेझैझोझौझंझःञञञाञिञीञुञूञृञेञैञोञौञंञःटटटाटिटीटुटूटृटेटैटोटौटंटःठठठाठिठीठुठूठृठेठैठोठौठंठःडडडाडिडीडुडूडृडेडैडोडौडंडःढढढाढिढीढुढूढृढेढैढोढौढंढःणणणाणिणीणुणूणृणेणैणोणौणंणःतततातितीतुतूतृतेतैतोतौतंतःथथथाथिथीथुथूथृथेथैथोथौथंथःदददादिदीदुदूदृदेदैदोदौदंदःधधधाधिधीधुधूधृधेधैधोधौधंधःनननानिनीनुनूनृनेनैनोनौनंनःपपपापिपीपुपूपृपेपैपोपौपंपःफफफाफिफीफुफूफृफेफैफोफौफंफःबबबाबिबीबुबूबृबेबैबोबौबंबःभभभाभिभीभुभूभृभेभैभोभौभंभःमममामिमीमुमूमृमेमैमोमौमंमःयययायियीयुयूयृयेयैयोयौयंयःरररारिरीरुरूरृरेरैरोरौरंरःलललालिलीलुलूलृलेलैलोलौलंलःवववाविवीवुवूवृवेवैवोवौवंवःशशशाशिशीशुशूशृशेशैशोशौशंशःषषषाषिषीषुषूषृषेषैषोषौषंषःसससासिसीसुसूसृसेसैसोसौसंसःहहहाहिहीहुहूहृहेहैहोहौहंहःळळळाळिळीळुळूळृळेळैळोळौळंळःक्षक्षक्षाक्षिक्षीक्षुक्षूक्षृक्षेक्षैक्षोक्षौक्षंक्षःज्ञज्ञज्ञाज्ञिज्ञीज्ञुज्ञूज्ञृज्ञेज्ञैज्ञोज्ञौज्" + "०१२३४५६७८९" + "|,!?")])
