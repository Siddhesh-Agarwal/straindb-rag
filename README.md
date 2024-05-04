# StrainDB RAG

An RAG Model based on [StrainsDB](https://strainsdb.org/). The Database for the project was provided by [Kenneth Reitz](https://github.com/kennethreitz).

___

## Steps involved

### Extracting and Validating the Data

- The SQLite DataBase consisted of 19 tables from which the `strains_strain` table was extracted.
  - **Solution**: extract data using SQLite module and create a `Strain` class (that inherits `pydantic.BaseModel`) for data validation
- Then the table's data cells consisted of lists in string format which needed to be converted back
  - **Solution**: `eval("[1, 2, 3]")` returns `[1, 2, 3]`
- The data needs to be saved.
  - **Solution**: dump data in JSON format.
 
### Data Tokenization

- The Data was embedded into ChromaDB (persistent client) using the `OpenAIEmbeddings` function.
