import { BPETokenizer } from "../src/tokenizer";
import * as path from "path";

const corpus = [
    `Pada zaman dahulu, hiduplah dua saudari tiri bernama Bawang Merah dan Bawang Putih. Sejak kecil, Bawang Putih sudah kehilangan ibunya.

Setelah itu, ayahnya menikah lagi dengan seorang wanita yang sudah memiliki anak bernama Bawang Merah. 

Sayangnya, tak lama kemudian, ayah Bawang Putih pun meninggal, meninggalkan Bawang Putih hidup bersama ibu tiri dan saudari tirinya. Sejak saat itu, kehidupan Bawang Putih berubah drastis. 

Setiap hari, dia dipaksa mengerjakan semua pekerjaan rumah, dari membersihkan rumah hingga mencuci pakaian, sementara Bawang Merah hanya duduk santai.

Suatu hari, ketika sedang mencuci di sungai, salah satu baju milik ibu tirinya terbawa arus dan hanyut. 

Bawang Putih sangat panik, tak tahu harus berbuat apa. Dalam kebingungannya, dia bertemu dengan seorang nenek tua yang baik hati.  

Nenek itu mengatakan bahwa dia telah menemukan baju yang hanyut, dan akan mengembalikannya.

Akan tetapi nenek punya satu permintaan, yakni Bawang Putih harus membantunya menyelesaikan beberapa pekerjaan rumah.

Dengan senang hati, Bawang Putih pun membantu nenek tersebut.

Setelah semua pekerjaan selesai, nenek itu menepati janjinya dan mengembalikan baju ibu tirinya.

Tak hanya itu, nenek itu juga memberinya hadiah, sebuah labu, dengan pilihan antara labu besar dan labu kecil.  Dengan rendah hati, Bawang Putih memilih labu yang kecil.

Setibanya di rumah, betapa terkejutnya Bawang Putih, ibu tiri, dan Bawang Merah ketika mereka membuka labu tersebut dan menemukan bahwa labu itu berisi perhiasan yang sangat banyak.

Tergiur dengan nasib baik Bawang Putih, keesokan harinya Bawang Merah meniru apa yang dilakukan oleh Bawang Putih.

Dia sengaja menghanyutkan bajunya dan mencari nenek yang sama. 

Ketika diberi pilihan, Bawang Merah dengan rakus memilih labu yang besar, berharap mendapatkan lebih banyak harta.

Namun, saat labu itu dibuka, isinya adalah ular-ular berbisa yang mengerikan.

Peristiwa ini menjadi pelajaran bagi Bawang Merah dan ibunya.

Mereka menyadari bahwa keserakahan dan perilaku buruk mereka terhadap Bawang Putih adalah sebuah kesalahan besar. 

Dengan hati yang penuh penyesalan, mereka meminta maaf kepada Bawang Putih atas semua yang telah mereka lakukan.

Pesan moral dari kisah ini yang bisa diajarkan ke anak adalah, jangan pernah bersikap buruk terhadap orang lain dan hindarilah sifat serakah.`,
    `Dahulu kala, hiduplah seorang wanita tua bernama Mbok Sirni yang tinggal seorang diri. 

Suaminya telah lama meninggal, dan meskipun setiap hari ia bekerja keras menanam sayur-mayur dan menjualnya di pasar, ia tetap merasa kesepian karena tak memiliki seorang anak pun. 

Setiap hari, Mbok Sirni memanjatkan doa kepada Tuhan, berharap agar diberikan seorang anak yang bisa menemani hari-harinya.

Suatu hari, ketika sedang berdoa dengan sungguh-sungguh, muncul raksasa bertubuh besar dan berwajah hijau, yang dikenal dengan nama Buto Ijo. 

Raksasa itu menawarkan bantuan kepada Mbok Sirni. “Aku bisa memberimu seorang anak, tetapi ada syaratnya. Ketika anak itu berusia enam tahun, kamu harus menyerahkannya kembali padaku,” ucap Buto Ijo dengan suara menggelegar.

Mbok Sirni, yang sangat merindukan seorang anak, langsung menyetujui permintaan itu tanpa berpikir panjang.

Buto Ijo kemudian memberikan benih mentimun kepada Mbok Sirni dan menyuruhnya menanamnya di ladang. 

Ia mengatakan bahwa di antara semua timun yang tumbuh, akan ada satu timun berwarna emas yang berisi bayi. Dua minggu kemudian, tanaman-tanaman timun mulai berbuah. 

Di antara semua buah, ada satu timun yang ukurannya paling besar dan berwarna emas. Dengan penuh kegembiraan, Mbok Sirni membelah timun itu dan di dalamnya terdapat seorang bayi perempuan yang cantik. 

Bayi itu kemudian diberi nama Timun Mas oleh Mbok Sirni. Waktu berlalu, Timun Mas tumbuh menjadi gadis yang cerdas dan baik hati. 

Mbok Sirni sangat menyayanginya. Namun, tibalah saat yang dinantikan Buto Ijo. Ia kembali untuk menagih janji dan mengambil Timun Mas.

Mbok Sirni yang tidak ingin kehilangan putri kesayangannya, berdoa dengan tulus agar Timun Mas bisa selamat.

Doanya dijawab oleh seorang petapa yang datang membawa empat benda: biji mentimun, jarum, garam, dan terasi.

Petapa itu memberikan benda-benda tersebut kepada Timun Maas, dengan pesan bahwa barang-barang itu bisa digunakan untuk melindungi dirinya.

Ketika Buto Ijo mulai mengejar Timun Maas, gadis itu melemparkan biji mentimun yang seketika tumbuh menjadi hutan mentimun yang lebat, menghalangi jalan Buto Ijo. 

Namun, Buto Ijo berhasil menerobos hutan itu dan terus mengejar. Lalu Timun Mas menaburkan jarum, yang berubah menjadi hutan bambu runcing yang menghadang Buto Ijo. 

Meski terluka, raksasa itu tetap tidak menyerah. Timun Mas kemudian menaburkan garam yang seketika berubah menjadi lautan luas. Namun, Buto Ijo tetap bisa menyeberang. 

Akhirnya, Timun Mas menaburkan terasi yang berubah menjadi lumpur panas. Buto Ijo terperangkap dan akhirnya tenggelam di dalam lumpur tersebut.

Dengan kematian Buto Ijo, Timun Maas akhirnya bebas dan kembali ke rumah. Ia hidup bahagia bersama Mbok Sirni tanpa lagi dihantui oleh ancaman raksasa yang jahat.

Pesan moral dari cerita ini adalah jangan pernah berniat jahat terhadap orang lain, karena pada akhirnya, kejahatan itu akan berbalik pada diri sendiri.`,
    `Pada suatu waktu di sebuah desa yang terpencil, hiduplah seorang janda tua bersama putri semata wayangnya yang bernama Darmi.

Meski Darmi memiliki wajah yang luar biasa cantik, sayangnya, perilakunya jauh dari indah. 

Ia dikenal sebagai gadis yang sangat egois dan tidak memiliki kepedulian terhadap orang lain, termasuk ibunya sendiri.

Darmi sangat terobsesi dengan penampilannya. Bahkan, setiap hari ia hanya sibuk mempercantik diri di dalam kamarnya, tanpa pernah mengurus atau membantu pekerjaan rumah. 

Kamarnya selalu berantakan, tapi Darmi tidak peduli. Baginya yang terpenting adalah wajahnya harus selalu tampil sempurna.

Sementara itu, ibunya yang sudah tua harus bekerja keras setiap hari untuk memenuhi kebutuhan hidup mereka.

Tidak peduli seberapa berat pekerjaannya, sang ibu selalu melakukannya dengan tulus demi memastikan Darmi bisa hidup nyaman.

Namun, meskipun begitu, Darmi seringkali memperlakukan ibunya dengan buruk. 

Ketika orang-orang bertanya siapa yang selalu berjalan di belakangnya, Darmi dengan kejam menyebut ibunya sebagai “budaknya.” Perlakuan kasar Darmi membuat hati ibunya hancur. 

Sang ibu yang tidak lagi sanggup menahan rasa sakit di hatinya, akhirnya berdoa dengan penuh kesedihan.

Tanpa disangka, doa itu membawa kutukan. Perlahan-lahan, tubuh Darmi mulai berubah menjadi batu. 

Saat proses itu terjadi, Darmi menangis dan memohon ampun kepada ibunya, tetapi semuanya sudah terlambat.

Akhirnya, Darmi berubah menjadi batu yang terus-menerus mengeluarkan air mata.

Cerita ini memberikan pelajaran berharga tentang pentingnya menghormati dan berbakti kepada orang tua.

Pesan moralnya adalah bahwa perilaku buruk dan ketidakpedulian terhadap orang tua bisa mendatangkan penyesalan yang tidak bisa diperbaiki.`
];

const tokenizer = new BPETokenizer({ vocabSize: 150, minFrequency: 2 });
tokenizer.train(corpus);
tokenizer.summary();

console.log("\n=== Encode & Decode ===\n");
for (const s of ["saya makan nasi goreng", "kami belajar musik", "mereka bermain basket"]) {
    const ids = tokenizer.encode(s);
    const tokens = ids.map(id => tokenizer.getToken(id));
    console.log(`"${s}" -> tokens: [${tokens.join(", ")}]`);
    console.log(`decoded: "${tokenizer.decode(ids)}"\n`);
}

console.log("=== Save & Load ===\n");
const vocabPath = path.join(__dirname, "..", "dataset", "bpe_vocab.json");
tokenizer.save(vocabPath);
const loaded = BPETokenizer.load(vocabPath);
const a = tokenizer.encode("saya belajar");
const b = loaded.encode("saya belajar");
console.log(`Konsisten: ${JSON.stringify(a) === JSON.stringify(b) ? "OK" : "FAIL"}`);

console.log("\n=== OOV Test ===\n");
for (const t of ["bermain piano", "belajar kimia", "matematikawan hebat"]) {
    const ids = tokenizer.encode(t);
    console.log(`"${t}" -> [${ids.map(id => tokenizer.getToken(id)).join(", ")}]`);
}
