This repo is a proposed solution to the **"Data Engineering Challenge"**  Titanic Kaggle competition with requirements updated by Artefact - School of data.


## Setup

First copy the `.env.sample` file to `.env` and fill in the values.

```bash
cp .env.sample .env
```

Then download your service account key from Google Cloud Platform and save it as `credentials/service-account.json`.

Once this is done and if you are using `pyenv-virtualenv` you can run the following command to setup the project.

```bash
make init_env
```

## Running the pipeline

To run the pipeline you can use the following command:
First and second CMD are there to process and train, third to check what is been uploaded to BQ.
1.
```bash
make preprocess
```

2.
```bash
make train
```

2.bis replace rf with any model key you want to use and that's referenced in our application. -help for more info
```bash
make train make ARGS="--model_key=rf"
```

3 (option).
```bash
make analysis
```
