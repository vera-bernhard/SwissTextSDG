# SwissTextSDG

## Connect GitHub to Drive and Colab
There seems to be two ways to work on a script from a GitHub repository from Colab.


1. **Access a specific .ipynb notebook (no access to the repo). This way makes most sense if you want to simply run an existing notebook without saving changes.**
   * Go to Colab and select `Open Colab` and then `GitHub`.
   * On the repo page, go to `Code` -> `local` -> `HTTPS`, and copy the link.
   * Paste the link in the field at the top of the window and click anywhere on the page to active the connection.
   * You will see a list of availalble notebooks.
     
2. **Clone the repo to Google Drive and push changes back to GitHub. This method allows you to work on the repo in the same manner as you would locally and gives full access to the data.**
   * Open any repo notebook from Colab as described in **Method 1**
   * On GitHub, go to `Settings` -> `Developer settings` -> `Personal access tokens` and create a new general token with permissions to work on repositories.
   * In the notebook opened on Colab, add following lines of code:
   ```
   git_token = '<your-access-token>'
   repo = '<repo name>'
   
   !git clone https://{git_token}@github.com/{repo-owner's-username}/{repo}.git
   ```
   * Copy the path to the cloned repository and navigate to the root directory.
   ```
   %cd <path-to-cloned-repo> # remember to add `%` before your commands in order to interact with Colab as if it were your terminal

   %pwd # make sure you are in the correct dir
   ```
   * At some point you will be prompted to identify yourself to push changes back to the repository, so you might as well add these lines of code at the top of the notebook:
   ```
   !git config --global user.email "<your-email>"
   !git config --global user.name "<your-username>"
   ```
   * You are all set now and can work exactly as you would do from your computer. In order to save the changes, use the usual commands, e.g.:
   ```
   !git add .
   !git commit -m 'cloned repo to Drive and connected via Colab'
   !git push origin main
   ```


Useful ressources:
* https://medium.com/analytics-vidhya/how-to-use-google-colab-with-github-via-google-drive-68efb23a42d
* https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb
