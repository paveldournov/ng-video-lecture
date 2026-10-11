"""
Download every work listed on Tolstoy's page at az.lib.ru and turn it into clean UTF-8 training text.

    python prepare_tolstoy.py                  # download (cached) and clean
    python prepare_tolstoy.py --download-only  # only fill the cache (~1.5 hours for all pages)
    python prepare_tolstoy.py --refresh        # re-download even if cached

Output:

    data/tolstoy/
        raw/<page>.html               original pages as downloaded (cache; delete to re-download)
        texts/<NN_section>/<file>.txt one clean text per work, one paragraph per line
        tolstoy_novels.txt            training files, works separated by a blank line:
        tolstoy_fiction.txt             novels / + novellas, stories, plays /
        tolstoy_all.txt                 everything Tolstoy wrote / works by other people about him
        about_tolstoy.txt
        manifest.json                 section, title, author, year, source URL, file and size of each work,
                                      and why a work was left out (stub page, duplicate, excluded)

Cleaning keeps only the work itself. The novels are trimmed with hand-checked start/end
markers (TRIM below); every other page is trimmed by rules that recognize publication
details at the top and editors' notes, commentaries, variants and indexes at the end.
Leftover HTML, footnote numbers, page numbers, links and e-mail addresses are removed,
scanning errors and pre-1918 spelling are normalized, and works that mostly repeat an
earlier one (e.g. reprints inside the complete-works volumes) are kept out of the
training files.
"""
import argparse
import html
import json
import re
import time
import urllib.request
from pathlib import Path
from urllib.parse import urljoin

BASE_URL = 'http://az.lib.ru/t/tolstoj_lew_nikolaewich/'
ABOUT_SECTIONS = {'Об авторе', 'О творчестве автора'} # memoirs and criticism by other people
OUT_DIR = Path('data/tolstoy')
REQUEST_DELAY = 1.0 # seconds between downloads, to be polite to the server
MIN_CHARACTERS = 1000 # pages with less text than this after cleaning are stubs (a link, a citation) and are skipped

# works listed without an author outside the "about" sections that are not (only) Tolstoy's writing; they go to
# about_tolstoy.txt instead of the Tolstoy training files
BY_OTHERS = {
    'text_1520': "memoirs of Tolstoyan peasants",
    'text_1898_v_kolonii_oldorfo': "report on a Tolstoyan colony",
    'text_1911_izvestiya_oldorfo': "Tolstoy museum society bulletin",
    'text_1912_ezhegodnik_oldorfo': "Tolstoy yearbook review",
    'text_1410': "letter to Tolstoy about Turgenev",
    'text_1900_pisma_alexeeva': "P. S. Alekseev's letters to Tolstoy",
    'text_1909_fr_krestnik': "article about Tolstoy's French godson",
    'text_1910_vosp_o_grote_oldorfo': "memoir of N. Ya. Grot",
    # two-sided correspondence: about half of the text is the other person's letters
    'text_0340': "correspondence with Leskov",
    'text_1560': "correspondence with Verigin",
    'text_1862_perepiska_s_botkinym_oldorfo': "correspondence with Botkin",
    'text_1894_perepiska_so_strahovym_oldorfo': "correspondence with Strakhov",
    'text_1896_perepiska_so_strahovym': "correspondence with Strakhov",
    'text_1906_perepiska_s_yagnom': "correspondence with Yagn",
}

# works listed under the section that are left out of the corpus, with the reason
EXCLUDE = {
    'text_0073': 'an earlier, shorter draft of War and Peace; it would duplicate most of volumes 1-4',
}

# where each work's own text starts and where the editorial material after it starts.
# the first paragraph matching `start` is kept; the first one after it matching `end` and everything after are dropped.
TRIM = {
    'text_0039':            dict(start=r'Часть первая'),
    'text_0040':            dict(start=r'ТОМ ПЕРВЫЙ', end=r'Примечания'),
    'text_0050':            dict(start=r'ТОМ ВТОРОЙ', end=r'Примечания'),
    'text_0060':            dict(start=r'ТОМ ТРЕТИЙ', end=r'Примечания'),
    'text_0070':            dict(start=r'ТОМ ЧЕТВЕРТЫЙ', end=r'ПЕЧАТНЫЕ ВАРИАНТЫ.*'),
    'text_0073':            dict(start=r'От автора'), # skip the publisher's preface, keep Tolstoy's own
    'text_0080':            dict(start=r'Мне отмщение, и аз воздам'),
    'text_0090':            dict(start=r'Часть первая', end=r'Комментарии\.?'),
    'text_1860_dekabristy': dict(start=r'Глава I', end=r'ПРИМЕЧАНИЯ'),
}

# Latin/Greek letters with stress marks that the source puts inside Russian words ("чт&#242;" = что) -> plain Cyrillic
STRESSED = str.maketrans('àáèéòóýόÀÁÈÉÒÓÝ', 'ааееооуоААЕЕООУ')
# letters that look the same in both alphabets; the scanned sources mix them up ("Bourіеnnе", "ХХІV.")
CYR_LOOKALIKE, LAT_LOOKALIKE = 'АВЕКМНОРСТХаеорсухіІ', 'ABEKMHOPCTXaeopcyxiI'
TO_LATIN = str.maketrans(CYR_LOOKALIKE, LAT_LOOKALIKE)
TO_CYRILLIC = str.maketrans(LAT_LOOKALIKE[:-2], CYR_LOOKALIKE[:-2]) # Latin i/I stay: Russian has no such letter
ROMAN_NUMERAL = re.compile(r'(?=.*[IVXІХ])[IVXLCDMІХСМ]+')
# pre-1918 spelling (many memoirs and articles about Tolstoy) -> modern letters; і and the final ъ are handled below
OLD_SPELLING = str.maketrans('ѣѢѳѲѵѴ', 'еЕфФиИ')

def normalize_word(word, old_spelling=False):
    word = word.translate(OLD_SPELLING)
    cyrillic_only = any(('а' <= c.lower() <= 'я' or c in 'ёЁ') and c not in CYR_LOOKALIKE for c in word)
    if cyrillic_only:
        word = word.translate(STRESSED) # stress marks only occur in Russian words
        word = re.sub(r'^ІІ(?=[а-яё])', 'П', word) # scanned "П" read as two "І": "ІІьер" -> "Пьер"
        word = word.replace('і', 'и').replace('І', 'И') # "міръ" -> "мир"
        # modern Russian never ends a word with ъ: in old spelling it is dropped ("домъ" -> "дом"),
        # in modern texts it is a scanning error for ь ("любитъ" -> "любить")
        if len(word) > 1 and word[-1] in 'ъЪ':
            word = word[:-1] if old_spelling else word[:-1] + ('ь' if word[-1] == 'ъ' else 'Ь')
    latin_only = any(c.isalpha() and not ('а' <= c.lower() <= 'я' or c in 'ёЁ') and c not in CYR_LOOKALIKE + LAT_LOOKALIKE for c in word)
    if latin_only and not cyrillic_only or not cyrillic_only and ROMAN_NUMERAL.fullmatch(word):
        return word.translate(TO_LATIN)
    if cyrillic_only and not latin_only:
        return word.translate(TO_CYRILLIC)
    return word

TRANSLIT: dict[str, str] = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
                    ['a', 'b', 'v', 'g', 'd', 'e', 'e', 'zh', 'z', 'i', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't',
                     'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya']))

def fetch(url, cache_path, refresh):
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes().decode('cp1251')
    print(f"  downloading {url}")
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (tolstoy corpus script)'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    cache_path.write_bytes(data)
    time.sleep(REQUEST_DELAY)
    return data.decode('cp1251')

def slugify(text):
    slug = ''.join(TRANSLIT.get(c, c) for c in text.lower())
    return re.sub(r'[^a-z0-9]+', '_', slug).strip('_')

def list_works(index_html):
    # the index groups works under headings like <a name=gr1>Романы:<gr0>; each work is one <DL><DT><li> entry
    parts = re.split(r'<a name=gr(\d+)>(.*?):<gr0>', index_html)
    if len(parts) < 4:
        raise SystemExit(f"no sections found on {BASE_URL}; the page layout may have changed")
    works = []
    for i in range(1, len(parts), 3):
        number, section, body = int(parts[i]), html.unescape(parts[i + 1]).strip(), parts[i + 2]
        for entry in re.findall(r'<DL><DT><li>(.*?)</DL>', body, re.S):
            match = re.search(r'<A HREF=([^ >]+?)\.shtml><b>(.*?)</b></A>(?:&nbsp;<small>\[(\d{4})[^\]]*\]</small>)?', entry)
            if not match:
                continue
            href, title, year = match.groups()
            # works by other people name their author before the link ("<b>Абрикосов В.А. </b> <A HREF=..."),
            # except in the sections about Tolstoy, where some only name the author inside the title
            author = re.search(r'<b>([^<]+?)\s*</b>\s*<A HREF', entry)
            page = href.strip('/').replace('/', '__') # cache name; relative links keep their plain name ("text_0039")
            by_tolstoy = author is None and section not in ABOUT_SECTIONS and page not in BY_OTHERS
            works.append(dict(
                section=section,
                section_dir=f"{number:02d}_{slugify(section)}",
                page=page,
                title=re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', '', title))).strip(' ,'),
                by_tolstoy=by_tolstoy,
                author='Толстой Л. Н.' if by_tolstoy else html.unescape(author.group(1)).strip() if author else None,
                year=int(year) if year else None,
                old_orthography='&#1122;' in entry, # the index marks pre-1918 spelling (ѣ, і, ъ) with a "Ѣ"
                url=urljoin(BASE_URL, href + '.shtml'),
            ))
    return works

def html_to_paragraphs(page_html, old_spelling=False):
    # the work itself sits between these markers; the rest of the page is site navigation, ratings, comments
    start = page_html.find('<!--Section Begins-->')
    end = page_html.rfind('<!--Section Ends-->')
    if start < 0 or end < 0:
        # older pages: the text follows a "Собственно произведение" comment and runs to the next comment
        start = page_html.find('Собственно произведение')
        start = page_html.find('-->', start) + 3 if start >= 0 else -1
        end = page_html.find('<!--', start) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError('text markers not found')
    body = page_html[start:end]

    # a "<" that doesn't open a tag must not be stripped as one (that would delete text up to the next ">"):
    # after a Russian letter it is a scanned stressed "а" ("пап<" = папа), anything else is literal text
    body = re.sub(r'(?<=[а-яА-Я])<(?![a-zA-Z/]|!--)', 'а', body)
    body = re.sub(r'<(?![a-zA-Z/]|!--)', '&lt;', body)
    body = re.sub(r'<!--.*?-->', ' ', body, flags=re.S)
    # plain-text pages (<pre>): a paragraph starts after a blank line or at an indented line; other line breaks are wrapping
    def pre_paragraphs(m):
        block = m.group(1).replace('\r', '')
        block = re.sub(r'\n[ \t]*\n|\n(?=[ \t])', '<br>', block)
        return block.replace('\n', ' ')
    body = re.sub(r'<pre\b[^>]*>(.*?)(?:</pre>|$)', pre_paragraphs, body, flags=re.S | re.I)
    body = re.sub(r'<sup>.*?</sup>', '', body, flags=re.S | re.I) # footnote numbers
    body = re.sub(r'<table.*?</table>', '\n', body, flags=re.S | re.I) # tables don't survive as plain text
    body = re.sub(r'<a\s[^>]*href="?mailto:.*?</a>', ' ', body, flags=re.S | re.I)
    body = re.sub(r'\s+', ' ', body) # source line breaks mean nothing; paragraphs come from the tags below
    body = re.sub(r'<(dd|p|br|div|li|h\d|tr|pre)\b[^>]*>|</(p|div|li|h\d|tr|pre)>', '\n', body, flags=re.I)
    body = re.sub(r'<[^>]+>', '', body) # any other tag: drop it, keep its text
    # some pages are escaped twice ("&amp;#948;") or show HTML as text ("&lt;DIV ALIGN=RIGHT&gt;")
    body = html.unescape(html.unescape(body)).replace('\xa0', ' ').replace('�', '')
    body = re.sub(r'</?[a-zA-Z][^<>\n]{0,40}>', '', body)
    old_spelling = old_spelling or body.count('ѣ') > 20 # the index doesn't flag every pre-1918 text

    paragraphs = []
    for line in body.split('\n'):
        line = re.sub(r'\{\d+\}', '', line) # page numbers of the printed edition, e.g. {123}
        # editors' restorations keep their text: "гр[аф]" -> "граф", "М[ихайловича?]" -> "Михайловича", "[в]" -> "в"
        line = re.sub(r'(?<=\w)\[(\w+)\??\]|(?<!\*)\[(\w{1,3})\]', lambda m: m.group(1) or m.group(2), line)
        # every other bracket is a translation or a footnote: "*[в форме фантазии]", "[Она еще красива? -- фр.]"
        line = re.sub(r'\s*\*?\[[^\]]*\]', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        line = re.sub(r'(?<=[^\W\d_])\*(?=[\s.,;:!?]|$)|\s\*$', '', line) # footnote asterisks: "уважение*." (not "***" or "188*")
        if not line or re.search(r'https?://|www\.|\S+@\S+\.\w+', line):
            continue
        line = re.sub(r'[^\W\d_]+', lambda m: normalize_word(m.group(), old_spelling), line)
        paragraphs.append(line)
    return paragraphs

BYLINE = re.compile(r'(Лев Николаевич Толстой|(Граф )?Лев Толстой|(Граф )?Л\. ?Н\. ?Толстой|Толстой Лев Николаевич)\.?', re.I)
# publication details at the top of a page: "Источник: ...", "Текст издания: Л.Н. Толстой. Собрание сочинений в 22 т. М.: ..."
SOURCE_LINE = re.compile(r'Источник|Оригинал (текста|здесь)|Текст издания|Текст предоставлен|номера страниц|Подчеркнутые слова'
                         r'|Собрание сочинений|Полное собрание|[Ии]здательств|\bИзд\.|ISBN|Тираж|\b(М|Л|СПб|Пг)\.: ?[А-ЯA-Z"«\d]'
                         r'|OCR|Электронн\w+ (публикац|издани|верси)|Подготовка текста|Перепечатка разрешается|Если Вы нашли ошибку|Распознавание'
                         r'|Воспроизводится по|Печатается по| // |Корректир|Редакция [А-Я]\.|Серия "|ACADEMIA|М\.-{1,2}Л\.'
                         r'|^"[^"]{2,60}", (\d{4}|кн\.|No|N|т\.)'
                         # title pages of the complete-works volumes
                         r'|Издание осуществляется|редакционной комиссии|редакторского комитета|под общей редакцией|Под общ\. ред\.'
                         r'|^\(?Издание:|^Серия \w+$|^ТОМ \d+$|^Редактор$'
                         r'|^[А-Я]\. ?[А-Я]\. [А-ЯЁ]+А$', re.I) # "В. Г. ЧЕРТКОВА"
# table of contents lines: "Редакционные пояснения ...................... XXXIX"
TOC_LINE = re.compile(r'.*(\.\s?){4,}\s*[\dIVXLC]+\.?')
# the complete-works volumes open with the editors' transcription rules: this heading and a run of long paragraphs
EDITORS_BLOCK = re.compile(r'(Редакционные пояснения|От редакции)\.?', re.I)
# where the editors' material after the text begins
NOTES_HEADING = re.compile(r'((Примечания|Комментари[ийя])( .{0,50})?|Печатные варианты.*|Варианты\b.*'
                           r'|Указатель\b.*|Редакционные пояснения\b.*|Список (сокращений|писем|источников|исправлений).*|Библиография|Сноски|Описание рукописей.*'
                           r'|Предисловие к .{0,40}тому.*|От редакции|История писания.*)\.?:?', re.I)
# editors' notes that start without a heading: '"Набег". -- Рассказ впервые опубликован в журнале...'
COMMENTARY_START = re.compile(r'"[^"]{2,120}"\.? -- .*(впервые|опубликован|напечатан)', re.I)
# what is left dangling at the very end: footnote numbers ("43 рок"), page numbers, separator lines
TRAILING_JUNK = re.compile(r'\d{1,4}(\s.{0,60})?|[-*_=~\s]{3,}')

def generic_trim(paragraphs):
    # pages from the tolstoy.ru edition open with a block of credits that ends with this line
    for i, p in enumerate(paragraphs[:80]):
        if p.startswith('Reproduction libre'):
            paragraphs = paragraphs[i + 1:]
            break
    paragraphs = [p for p in paragraphs if not TOC_LINE.fullmatch(p)]
    i = 0
    while i < min(len(paragraphs), len(paragraphs) // 7 + 30):
        if EDITORS_BLOCK.fullmatch(paragraphs[i]):
            j = i + 1
            # the rules are sentences; the block ends at the next heading (short, no final period)
            while j < len(paragraphs) and (len(paragraphs[j]) >= 60 or paragraphs[j].endswith('.') and not paragraphs[j].isupper()):
                j += 1
            paragraphs = paragraphs[:i] + paragraphs[j:]
        else:
            i += 1
    # a contents list near the top: the heading and the run of short title lines after it
    for i, p in enumerate(paragraphs[:len(paragraphs) // 7 + 30]):
        if re.fullmatch(r'(С ?О ?Д ?Е ?Р ?Ж ?А ?Н ?И ?Е|О ?Г ?Л ?А ?В ?Л ?Е ?Н ?И ?Е)\.?', p, re.I): # also "С О Д Е Р Ж А Н И Е"
            j = i + 1
            while j < len(paragraphs) and len(paragraphs[j]) < 150:
                j += 1
            paragraphs = paragraphs[:i] + paragraphs[j:]
            break
    front = 25
    head = [p for p in paragraphs[:front] if not (BYLINE.fullmatch(p) or len(p) < 250 and SOURCE_LINE.search(p))]
    paragraphs = head + paragraphs[front:]
    # notes headings near the start are part of the work (e.g. an author's own preface notes), so only look past the first 20%
    cut = next((i for i, p in enumerate(paragraphs) if i > len(paragraphs) * 0.2
                and (len(p) < 100 and NOTES_HEADING.fullmatch(p) or COMMENTARY_START.match(p))), None)
    paragraphs = paragraphs[:cut]
    while paragraphs and (TRAILING_JUNK.fullmatch(paragraphs[-1]) or len(paragraphs[-1]) < 250 and SOURCE_LINE.search(paragraphs[-1])):
        paragraphs.pop()
    return paragraphs

def trim(paragraphs, page):
    rules = TRIM.get(page)
    if rules is None:
        return generic_trim(paragraphs)
    matches = lambda pattern, p: re.fullmatch(pattern, p, flags=re.I) is not None
    starts = [i for i, p in enumerate(paragraphs) if matches(rules['start'], p)]
    if not starts:
        raise ValueError(f"start marker {rules['start']!r} not found")
    paragraphs = paragraphs[starts[0]:]
    if 'end' in rules:
        ends = [i for i, p in enumerate(paragraphs) if matches(rules['end'], p)]
        if not ends:
            raise ValueError(f"end marker {rules['end']!r} not found")
        paragraphs = paragraphs[:ends[0]]
    return paragraphs

def file_name(work):
    return f"{work['year']}_{slugify(work['title'])[:60].rstrip('_')}.txt"

def paragraph_key(p):
    return re.sub(r'\W+', '', p.lower())

# training files built from the cleaned works; duplicates and EXCLUDE entries are left out of all of them
CORPORA = {
    'tolstoy_novels.txt': lambda w: w['section'] == 'Романы',
    'tolstoy_fiction.txt': lambda w: w['by_tolstoy'] and w['section_dir'][:2] <= '06', # novels, novellas, stories, plays
    'tolstoy_all.txt': lambda w: w['by_tolstoy'],
    'about_tolstoy.txt': lambda w: not w['by_tolstoy'], # memoirs and criticism by other people
}

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--refresh', action='store_true', help='re-download pages even if they are cached')
    parser.add_argument('--download-only', action='store_true', help='only fill the raw/ cache, do not clean')
    args = parser.parse_args()

    raw_dir, texts_dir = OUT_DIR / 'raw', OUT_DIR / 'texts'
    raw_dir.mkdir(parents=True, exist_ok=True)

    works = list_works(fetch(BASE_URL, raw_dir / 'index.html', args.refresh))
    print(f"found {len(works)} works in {len({w['section'] for w in works})} sections")
    if args.download_only:
        failed = []
        for n, work in enumerate(works, 1):
            try:
                fetch(work['url'], raw_dir / f"{work['page']}.html", args.refresh)
            except Exception as e:
                failed.append(work['url'])
                print(f"  FAILED {work['url']}: {e}")
            if n % 50 == 0:
                print(f"{n}/{len(works)} pages")
        print(f"done: {len(works) - len(failed)} pages cached, {len(failed)} failed")
        return

    manifest, texts, used_paths = [], {}, set()
    seen = {True: set(), False: set()} # long paragraphs already kept, separately for Tolstoy's works and the others
    for work in works:
        entry = dict(work, file=None, paragraphs=0, characters=0, in_corpus=False, note=None)
        manifest.append(entry)
        if work['page'] in EXCLUDE:
            entry['note'] = f"excluded: {EXCLUDE[work['page']]}"
            continue
        try:
            page_html = fetch(work['url'], raw_dir / f"{work['page']}.html", args.refresh)
            paragraphs = trim(html_to_paragraphs(page_html, work['old_orthography']), work['page'])
        except Exception as e:
            entry['note'] = f"failed: {e}"
            print(f"  FAILED {work['title']} ({work['url']}): {e}")
            continue
        if sum(len(p) for p in paragraphs) < MIN_CHARACTERS:
            entry['note'] = 'too short: the page only holds a link or a bibliographic note'
            continue

        # the complete-works volumes and some sections reprint texts found elsewhere on the page
        long_keys = [paragraph_key(p) for p in paragraphs if len(p) >= 200]
        repeated = sum(k in seen[work['by_tolstoy']] for k in long_keys) / max(len(long_keys), 1)
        if repeated > 0.5:
            entry['note'] = f"duplicate: {repeated:.0%} of its long paragraphs appear in earlier works"
        else:
            seen[work['by_tolstoy']].update(long_keys)
            entry['in_corpus'] = True

        path = texts_dir / work['section_dir'] / file_name(work)
        n = 2
        while path in used_paths: # same year and title in one section
            path = path.with_name(f"{path.stem.rsplit('__', 1)[0]}__{n}.txt")
            n += 1
        used_paths.add(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = '\n'.join(paragraphs) + '\n'
        path.write_text(text, encoding='utf-8')
        texts[work['page']] = text
        entry.update(file=path.as_posix(), paragraphs=len(paragraphs), characters=len(text))

    for name, include in CORPORA.items():
        chosen = [m for m in manifest if m['in_corpus'] and include(m)]
        (OUT_DIR / name).write_text('\n'.join(texts[m['page']] for m in chosen), encoding='utf-8')
        print(f"{name:20} {len(chosen):4} works {sum(m['characters'] for m in chosen):>12,} characters")
    (OUT_DIR / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    problems = [m for m in manifest if m['note'] and not m['note'].startswith('excluded')]
    print(f"{sum(m['file'] is not None for m in manifest)} works written to {texts_dir.as_posix()}/; {len(problems)} skipped or duplicate (see manifest.json):")
    for m in problems:
        print(f"  {m['title'][:60]:60} {m['note']}")

if __name__ == '__main__':
    main()
