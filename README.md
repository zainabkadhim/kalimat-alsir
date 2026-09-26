# Kalimat Al-Sir (كلمة السر)

**Kalimat Al-Sir** is an interactive Arabic word-association game inspired by games like *CONTEXTO*. The goal is simple: guess the hidden secret word by entering Arabic words and using semantic similarity scores to guide your guesses.

Rather than checking letter positions, the game uses NLP vector embeddings to calculate how close your guess is to the secret word in meaning and usage context.

---

## 🎮 Play the Game

You can play the live app here:  
👉[Play Kalimat Al-Sir](https://kalimat-alsir.streamlit.app/)

---

## 🕹️ How to Play

1. Enter an Arabic word as your guess.
2. The game calculates a similarity score between your guess and the secret target word using **FastText** Arabic embeddings.
3. Your guesses are ordered by similarity so you can see whether you are getting warmer or colder.
4. Keep narrowing down the context until you hit 100% similarity and reveal the secret word!
