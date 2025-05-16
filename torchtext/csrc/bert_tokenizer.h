#include <torchtext/csrc/export.h>
#include <torchtext/csrc/vocab.h>
#include <string>
#include <vector>

namespace torchtext {

//typedef std::basic_string<uint32_t> UString;
typedef std::u32string UString; //fix compliation on xcode 16 due to the missing of std::char_traits<unsigned int>, refer to https://github.com/pytorch/text/pull/2292/commits/059d2f0501de2fe7db6cb2e4becbf0a3912ec4c4 and https://github.com/pytorch/text/issues/2291
typedef ska_ordered::order_preserving_flat_hash_map<std::string, int64_t>
    IndexDict;

// stores (do_lower_case, strip_accents, never_split, list of tokens in
// vocabulary)
typedef std::tuple<
    bool,
    c10::optional<bool>,
    std::vector<std::string>,
    std::vector<std::string>>
    BERTEncoderStates;

struct BERTEncoder : torch::CustomClassHolder {
  TORCHTEXT_API BERTEncoder(
      const std::string& vocab_file,
      bool do_lower_case,
      c10::optional<bool> strip_accents,
      std::vector<std::string> never_split);
  BERTEncoder(
      Vocab vocab,
      bool do_lower_case,
      c10::optional<bool> strip_accents,
      std::vector<std::string> never_split);
  TORCHTEXT_API std::vector<std::string> Tokenize(std::string text);
  TORCHTEXT_API std::vector<int64_t> Encode(std::string text);
  TORCHTEXT_API std::vector<std::vector<std::string>> BatchTokenize(
      std::vector<std::string> text);
  TORCHTEXT_API std::vector<std::vector<int64_t>> BatchEncode(
      std::vector<std::string> text);

  Vocab vocab_;
  bool do_lower_case_;
  c10::optional<bool> strip_accents_ = {};
  std::vector<std::string> never_split_;
  std::set<std::string> never_split_set_;

 protected:
  UString _clean(
      const UString& text,
      bool strip_accents,
      bool is_never_split_token);
  void _max_seg(const std::string& s, std::vector<std::string>& results);
  UString _basic_tokenize(const UString& token, bool is_never_split_token);
  void split_(
      const std::string& str,
      std::vector<std::string>& tokens,
      const char& delimiter = ' ');
  static std::string kUnkToken;
};

TORCHTEXT_API BERTEncoderStates
_serialize_bert_encoder(const c10::intrusive_ptr<BERTEncoder>& self);
TORCHTEXT_API c10::intrusive_ptr<BERTEncoder> _deserialize_bert_encoder(
    BERTEncoderStates states);
} // namespace torchtext
