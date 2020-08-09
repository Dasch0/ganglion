use ferris_says;
use std;

fn main() {
  let stdout = std::io::stdout();
  let message = String::from("Hello, I'm a crab!");
  let width = message.chars().count();

  let mut writer = std::io::BufWriter::new(stdout.lock());
  ferris_says::say(message.as_bytes(), width, &mut writer).unwrap();
}

