# LaTeX Project

Dieses Projekt enthält alle meine LaTeX-Dateien.

## Kompilieren der LaTeX-Dateien

Um die LaTeX-Dateien zu kompilieren, folge diesen Schritten:

1. **Erstelle das PDF-Dokument:**

   ```bash
   pdflatex -output-directory=./out main.tex
   ```

2. **Führe Biber aus:**

   ```bash
   biber ./out/main
   ```

3. **Erstelle die Glossare:**

   ```bash
   makeglossaries -d ./out main
   ```

4. **Kompiliere das Dokument erneut:**

   ```bash
   pdflatex -output-directory=./out main.tex
   pdflatex -output-directory=./out main.tex
   ```

## Struktur

```
latex_project/
├── main.tex
├── out/
├── bib/
│ └── references.bib
├── glossaries/
│ └── glossary.tex
└── README.md
```

# compile

```bash
latexmk -pdf -outdir=./out main.tex
biber ./out/main
makeglossaries -d ./out main
latexmk -pdf -outdir=./out main.tex
latexmk -pdf -outdir=./out main.tex
```

```bash
latexmk -pdf -outdir=./out main.tex
bibtex ./out/main
makeglossaries -d ./out main
latexmk -pdf -outdir=./out main.tex
latexmk -pdf -outdir=./out main.tex
```
