import socket
import numpy as np

# MWE Extractor
language = 'en'
focus_token = 'time'
number_of_10k_results = 5
n_random_analyses = 2
evaluation_size = 1000
min_freq = 2

csv_path = 'data/' + language + '/00_csvs/' + focus_token + '/'
ngrams_path = 'data/' + language + '/01_ngrams/' + focus_token + '/'
results_path = 'data/' + language + '/02_results/' + focus_token + '_' + str(number_of_10k_results) + '/'
aggregate_matrix_path = ngrams_path + focus_token + '_aggregate_matrix_min_' + str(min_freq) + '_' + str(number_of_10k_results) + '.npy'
subgram_path = ngrams_path + focus_token + '_subgram_freqs_min_' + str(min_freq) + '_' + str(number_of_10k_results) + '.tsv'
token_path = ngrams_path + focus_token + '_token_freqs_min_' + str(min_freq) + '_' + str(number_of_10k_results) +'.tsv'

left_window_size = 5
right_window_size = 5
left_indices = [[5], [4, 5], [3, 4, 5], [2, 3, 4, 5], [1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
right_indices = [[], [6], [6, 7], [6, 7, 8], [6, 7, 8, 9], [6, 7, 8, 9, 10]]
prefixes = ['L5', 'L4', 'L3', 'L2', 'L1', 'KW', 'R1', 'R2', 'R3', 'R4', 'R5']
length_matrix = np.array([[1, 2, 3, 4, 5, 6],
				 [2, 3, 4, 5, 6, 7],
				 [3, 4, 5, 6, 7, 8],
				 [4, 5, 6, 7, 8,  9],
				 [5, 6, 7, 8, 9, 10],
				 [6, 7, 8, 9, 10, 11]])



# 99_pipeline.py
computer_name = socket.gethostname()
root = 'data/sentences/'
raw_path = root + 'raw_files/'
valid_path = root + 'valid_sentences/'
analysis_fst = '/usr/bin/flookup -b -x /home/lubuntu0/TRmorph/trmorph.fst'
generation_fst = '/usr/bin/flookup -i -b -x /home/lubuntu0/TRmorph/trmorph.fst'
segmentation_fst = '/usr/bin/flookup -b -x /home/lubuntu0/TRmorph/segment.fst'
morph_store_root = '/home/lubuntu0/TRmorph/morph_store/'

replacements = {'Å  ': 'ş', 'Ä ± ': 'ı', 'Ä  ': 'ğ', 'Ã¼': 'ü', 'þ': 'ş', 'ý': 'ı', 'ð': 'ğ', '�': '"', 'Þ': 'Ş',
				'Ã  ': 'Ö', 'Ý': 'İ', ' ¬ ': '', '’': '\'', '‘': '\'', '′': '\'', '΄': '\'', '`': '\'', '´': '\'',
				'”': '"', '“': '"', '»': '"', '«': '"', '„': '"', '″': '"', '‟': '"', '\'\'': '"', '­': '-', '‒': '-',
				'–': '-', '—': '-', '•': '●', '·': '●', '★': '●', '►': '●'}

abbreviations = ['Abs', 'acc', 'ad', 'Ad', 'al', 'Al', 'An', 'Anl', 'Apt', 'apt', 'APT', 'ark', 'Art', 'AS', 'AŞ',
				 'assy',
				 'Aufl', 'Av', 'AV', 'Ave', 'Bağ', 'BAĞ', 'bak', 'Bilg', 'BK', 'bk', 'Bk', 'bkz', 'Bkz', 'Bl', 'Bld',
				 'bld', 'Bldg',
				 'Blk', 'Blv', 'Böl', 'Brg', 'Bros', 'Bşk', 'Bul', 'Bulv', 'BV', 'Cad', 'cad', 'CAD', 'cadd', 'cc',
				 'Cd',
				 'cd', 'Ch', 'cm', 'Co', 'CO', 'Col', 'comp', 'Cor', 'Corp', 'Cum', 'Dağ', 'dah', 'dak', 'Dan', 'DAN',
				 'Dem', 'Den', 'Dept', 'dept', 'Dia', 'diğ', 'Dil', 'dk', 'dn', 'Doç', 'DOÇ', 'Dr', 'DR', 'Ea', 'ea',
				 'Ecz', 'Eğt', 'Ek', 'EK', 'ELEK', 'Elek', 'Elk', 'Esq', 'etc', 'excl', 'Faal', 'Fab', 'FABR', 'ft',
				 'fwd', 'Fwd', 'Gen', 'GID', 'Gn', 'gr', 'Gr', 'Güv', 'Hak', 'Haz', 'HD', 'Hiz', 'HİZ', 'Hizm', 'HK',
				 'hk', 'Hz', 'İhr', 'İKT', 'Inc', 'INC', 'Ind', 'ins', 'inş', 'İS', 'İth', 'Jr', 'Jud', 'Kat', 'Kdz',
				 'kg', 'KG', 'Kim', 'KİM', 'Km', 'km', 'Ko', 'Koll', 'Kont', 'Koop', 'Kozm', 'Kur', 'Lab', 'lbs', 'lit',
				 'LLC', 'LLP', 'lt', 'Ltd', 'LTD', 'ltd', 'LTDA', 'mad', 'Mad', 'Mah', 'mah', 'MAH', 'mak', 'maks',
				 'Maks', 'Mal', 'Malz', 'MAR', 'Marj', 'max', 'MAX', 'Max', 'md', 'Md', 'Mer', 'Merk', 'Met', 'MG',
				 'mg', 'Mh', 'mh', 'Mim', 'min', 'Min', 'MİN', 'Mkz', 'mlz', 'mm', 'Mr', 'Mrk', 'Mrs', 'Ms', 'mt', 'Mt',
				 'Muh', 'Müh', 'müh', 'MÜH', 'MÜK', 'Mük', 'Nak', 'NAK', 'Nakl', 'Nav', 'No', 'no', 'NO', 'Nr', 'nr',
				 'Op',
				 'opr', 'ör', 'Ord', 'Ore', 'Org', 'org', 'ORG', 'örn', 'Örn', 'Ort', 'ort', 'PAR', 'par', 'Par', 'Pat',
				 'Paz', 'PAZ', 'pg', 'Ph', 'PK', 'PLC', 'Plc', 'pp', 'Prg', 'prg', 'Prof', 'PROF', 'Pte', 'Pty', 'Pvt',
				 'ref', 'REF', 'Ref', 'reg', 'Rek', 'Rep', 'Rev', 'REV', 'SA', 'Sa', 'San', 'SAN', 'Say', 'Sdn', 'Sek',
				 'Sept', 'seq', 'sf', 'sh', 'sic', 'Sk', 'SK', 'sk', 'Sn', 'SN', 'sn', 'Sok', 'sok', 'SOK', 'Sp', 'sqm',
				 'Srv', 'ss', 'St', 'st', 'Stg', 'stg', 'Sti', 'Şti', 'str', 'Str', 'Sttr', 'Şub', 'sy', 'Syf', 'Taah',
				 'TAAH', 'taah', 'Tah', 'Taks', 'Tar', 'Tas', 'Taş', 'TAŞ', 'TC', 'Tek', 'Tekn', 'Teks', 'Tel', 'tel',
				 'Tem', 'Tes', 'TES', 'Tic', 'tic', 'TİC', 'Tl', 'Top', 'TUR', 'Tur', 'Turz', 'Univ', 'Ür', 'Uy', 'VAD',
				 'vb', 'VB', 'vd', 'Vek', 'VNo', 'vo', 'Vol', 'vs', 'VS', 'Vs', 'www', 'Yard', 'Yat', 'YAZ', 'Yön',
				 'Yrd',
				 'YRD', 'Yük', 'Yun', 'Yzb', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII',
				 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XI', 'i', 'ii', 'iii', 'iv', 'v', 'vi',
				 'vii',
				 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx', 'xi']

tag_descriptions = {'AB': ' abbreviation', 'AL': ' all lowercase', 'AM': ' ampersand', 'AP': ' apostrophe',
					'AS': ' asterisk',
					'AT': ' at sign', 'AU': ' all uppercase', 'BL': ' bullet', 'BS': ' backslash', 'CL': ' column',
					'CM': ' comma', 'CP': ' closing parenthesis', 'DA': ' dash', 'DG': ' degree', 'DL': ' dollar sign',
					'DT': ' dot', 'EL': ' ellipsis', 'EM': ' empty', 'EU': ' Euro sign', 'EX': ' exclamation mark',
					'FP': ' floating point number', 'FS': ' forward slash', 'HA': ' hash', 'HW': ' hyphenated word',
					'IN': ' initialism', 'IS': ' integer sequence', 'IU': ' initial letter uppercase',
					'MC': ' mixed case',
					'MO': ' mathematical operator', 'OP': ' opening parenthesis', 'PC': ' percent sign', 'PI': ' pipe',
					'QT': ' quotation mark', 'QU': ' question mark', 'RE': ' registered', 'RT': ' Ratio / Time',
					'SB': 'sentence boundary', 'SC': ' semicolon', 'SN': ' separated numbers', 'TM': ' trademark sign',
					'UR': ' URL', 'US': ' underscore'}

regex_db = {r'[a-zçığöşüâîû]+$': 'AL', r'[A-ZÇİĞÖŞÜÂÎÛ][a-zçığöşüâîû]+$': 'IU', r'\,$': 'CM', r'\.$': 'DT',
			r'\<\/s\>\<s\>$': 'SB', r'[0-9]+$': 'IS', r'"$': 'QT', r'\'$': 'AP', r'$': 'EM', r'[A-ZÇİĞÖŞÜÂÎÛ]+$': 'AU',
			r'\…+$': 'EL', r'\.\.+$': 'EL', r'\?+$': 'QU', r'\:$': 'CL', r'[\)\]\}]$': 'CP', r'[\(\[\{]$': 'OP',
			r'\;$': 'SC', r'[A-ZÇİĞÖŞÜÂÎÛa-zçığöşüâîû]+\-[A-ZÇİĞÖŞÜÂÎÛa-zçığöşüâîû]+': 'HW',
			r'([0-9]+\.)+[0-9]*$': 'SN', r'[0-9]*[\.,][0-9]+$': 'FP', r'\/$': 'FS',
			r'([A-ZÇİĞÖŞÜÂÎÛa-zçığöşüâîû]\.)+[A-ZÇİĞÖŞÜÂÎÛa-zçığöşüâîû]*$': 'IN', r'\![\!\?]*$': 'EX', r'\-$': 'DA',
			r'\*+$': 'AS',
			r'(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)': 'UR',
			r'(?=.*[a-zçığöşüâîû])(?=.*[A-ZÇİĞÖŞÜÂÎÛ])[a-zçığöşüâîûA-ZÇİĞÖŞÜÂÎÛ]+$': 'MC', r'[\•\●\·\★\►]$': 'BL',
			r'\@$': 'AT', r'[A-ZÇİĞÖŞÜÂÎÛ][a-zçığöşüâîû]+\.$': 'AB', r'\%$': 'PC', r'[\=\+\×\<\>]$': 'MO', r'\°$': 'DG',
			r'\&$': 'AM', r'\\$': 'BS', r'\#$': 'HA', r'\|$': 'PI', r'\_$': 'US', r'\®$': 'RE', r'\€$': 'EU',
			r'\$$': 'DL',
			r'\™$': 'TM', r'[0-9]+\:[0-9]+$': 'RT'}

categories = {
	None: '', 'EM': '', 'AP': '', 'QT': '', 'OP': '', 'CP': '',
	'SB': '|',
	'AS': 'B', 'BL': 'B',
	'IU': 'W', 'AL': 'W', 'MC': 'W', 'IS': 'W', 'FP': 'W', 'WA': 'W', 'SN': 'W', 'AU': 'W', 'HW': 'W',
	'IN': 'W', 'AB': 'W', 'RT': 'W', 'UR': 'W',
	'CM': 'M', 'SC': 'M', 'FS': 'M', 'DA': 'M', 'RE': 'M', 'EU': 'M', 'DL': 'M', 'TM': 'M',
	'MO': 'M', 'DG': 'M', 'AM': 'M', 'PC': 'M', 'AT': 'M', 'HA': 'M', 'PI': 'M', 'US': 'M', 'BS': 'M',
	'DT': 'E', 'EX': 'E', 'QU': 'E', 'EL': 'E', 'CL': 'E'
}

removable_morphs = ['â', 'î', 'û', '<0>', '<cpl:pres>', '<N:prop>', '<Num:rom>', ':mredup>', '<si><Adj>', ':typo',
					'cv:ye', '<part:past><la>', '<vn:past><la>', '<Sym>', '<Num:time>', 'e<Ij>']

non_surface_tags = ['<N>', '<Adj>', '<0>', '<V>', '<3s>', '<Adv>', '<Advj>', '<Prn>', '<Cnj:adv>', '<Postp:adv:nomC>']

# NOTE: Bundle possessive markers only (except p3s!)
tag_bundles = {'<p1s>': '<poss>', '<p2s>': '<poss>', '<p1p>': '<poss>', '<p2p>': '<poss>', '<p3p>': '<poss>',
			   '<part:pres>': '<An>', '<vn:pres>': '<An>', '<part:past>': '<DHk>', '<vn:past>': '<DHk>',
			   '<part:fut>': '<AcAk>', '<vn:fut>': '<AcAk>', '<Adj>': '<Advj>', '<Adv>': '<Advj>', '<yis>': '<iş>',
			   '<vn:yis>': '<iş>',
			   '<vn:infmAK>': '<mAk>', '<vn:infmA>': '<mA>', '<cv:erek>': '<ArAk>'}

# tag_bundles = {'<cont>': '', '<fut>': '', '<past>': '', '<imp>': '', '<aor>': '',
#                '<cond>': '', '<evid>': '', '<obl>': '', '<impf>': '', '<abil>': '',
#                '<cpl:evid>': '', '<cpl:past>': '', '<cpl:cond>': '', '<dir>': '', '<opt>': '',
#                '<1s>': '', '<2s>': '', '<3s>': '', '<1p>': '', '<2p>': '', '<3p>': '',
#                '<p1s>': '<poss>', '<p2s>': '<poss>', '<p1p>': '<poss>', '<p2p>': '<poss>', '<p3p>': '<poss>',
#                '<part:pres>': '<An>', '<vn:pres>': '<An>', '<part:past>': '<DHk>', '<vn:past>': '<DHk>',
#                '<part:fut>': '<AcAk>', '<vn:fut>': '<AcAk>', '<Adj>': '<Advj>', '<Adv>': '<Advj>', '<yis>': '<iş>', '<vn:yis>': '<iş>',
#                '<vn:infmAK>': '<mAk>', '<vn:infmA>': '<mA>', '<cv:erek>': '<ArAk>'}

vstem_tags = ['<la>', '<lan>', '<las>', '<caus>', '<pass>', '<rcp>', '<rfl>', '<tamp>']

function_words = ['ve', 'bir', 'bu', 'da', 'de', 'için', 'ile', 'çok', 'olarak', 'daha', 'olan', 'gibi', 'en', 'her',
				  'o', 'ne', 'kadar', 'ama', 'sonra', 'ise', 'ya', 'ki', 'var', 'ilk', 'zaman', 'ben', 'değil', 'son',
				  'göre', 'veya', 'ancak', 'tarafından', 'önce', 'diye', 'içinde', 'tüm', 'kendi', 'aynı', 'ilgili',
				  'sadece', 'hem', 'yok', 'şekilde', 'diğer', 'arasında', 'bile', 'karşı', 'hiç', 'nasıl', 'tek', 'şey',
				  'fazla', 'birlikte', 'böyle', 'bunun', 'başka', 'bütün', 'çünkü', 'yani', 'bunu', 'şu', 'biz', 'bazı',
				  'yine', 'ortaya', 'artık', 'üzerine', 'mi', 'benim', 'onun', 'üzerinde', 'neden', 'biri',
				  'ayrıca', 'tam', 'üzere', 'özellikle', 'az', 'şimdi', 'bizim', 'yerine', 'bugün', 'hiçbir', 'eğer',
				  'onu', 'fakat', 'burada', 'hakkında', 'konusunda', 'bana', 'biraz', 'hemen', 'kez', 'ardından',
				  'işte',
				  'ikinci', 'uygun', 'zaten', 'birçok', 'pek', 'üç', 'an', 'buna', 'değildir', 'rağmen', 'herşey',
				  'altında', 'hatta', 'aslında', 'öyle', 'şöyle', 'geri', 'süre', 'yeniden', 'milyon', 'ona', 'ta',
				  'hep', 'zamanda', 'ay', 'belki', 'gerek', 'nedeniyle', 'beni', 'yakın', 'içerisinde', 'bağlı', 'bize',
				  'mümkün', 'tekrar', 'gereken', 'konusu', 'konuda', 'anda', 'hafta', 'bundan', 'bunlar', 'size',
				  'dışında', 'dolayı', 'çeşitli', 'hangi', 'yönelik', 'siz', 'belli', 'sen', 'ait', 'yaklaşık',
				  'gerçekten', 'oldukça', 'herhangi', 'bi', 'kendini', 'sürekli', 'sonunda', 'türlü', 'birkaç', 'ayrı',
				  'onlar', 'ana', 'yanında', '9', 'iç', 'açısından', 'herkes', 'arada', 'gerekli', 'boyunca', 'sizin',
				  'onların', 'adına', 'evet', 'nedenle', 'tür', 'hale', 'üst', 'yerde', 'bunların', 'haline', 'yüzden',
				  'kalan', 'hala', 'şeklinde', 'sonucu', 'kimse', 'halinde', 'arasındaki', 'kendisine', 'ilişkin',
				  'başına', 'böylece', 'yere', 'üzerinden', 'tamamen', 'sırasında', 'mevcut', 'on', 'mu', 'beraber',
				  'derece', 'içine', 'bizi', 'halde', 'yana', 'onları', 'yanı', 'sık', 'dört', 'şeyler',
				  'bence', 'adet', 'yandan', 'dış', 'dolayısıyla', 'beri', 'bazen', 'bunları', 'önceki', 'kim',
				  'örneğin',
				  'orada', 'karşısında', 'önünde', 'ileri', 'henüz', 'sayesinde', 'kaç', 'itibaren', 'tabii', 'altına',
				  'kendisi', 'onlara', 'çoğu', 'başında', 'zorunda', 'ön', 'kendine', 'birinci', 'şeyi', 'mutlaka',
				  'dün',
				  'kendisini', 'alt', 'başta', 'lazım', 'amacıyla', 'defa', 'dahil', 'peki', 'sonrası', 'senin', 'sizi',
				  'önüne', 'araya', 'sana', 'benzer', 'sanki', 'kapsamında', 'dair', 'sırada', 'anlamda', 'dakika',
				  'biçimde', 'herkesin', 'nedir', 'öncelikle', 'yalnız', 'sayıda', 'elbette', 'asla', 'asıl', 'dahi',
				  'sonraki', 'ister', 'tane', 'öne', 'ara', 'genellikle', 'kendilerine', 'arası', 'idi', 'mesela',
				  'yerinde', 'sonrasında', 'seni', 'bende', 'üçüncü', 'hepsi', 'hayır', 'oysa', 'acaba', 'kesinlikle',
				  'neler', 'birisi', 'çerçevesinde', 'açıdan', 'doğrultusunda', 'yönünde', 'esnasında', 'halini']

breaking_tokens = ['</s><s>', '.', ',', ':', ')', '(', '…', ';', '!', '?', '"', '""', '-', '..', '...', '....',
					'[', ']', '?!', '???', '●', '*']

english_separables = ',.:()?!;]['
