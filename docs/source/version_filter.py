import os

def create_version_filter(include: str): 
  # Create a negative-lookahead regexp to exclude the version from being generated
  def to_rexp(dir): 
    return "(?!" + dir.replace('.', '\.') + ")"
  
  versions = os.listdir('../../dowhy-docs')
  versions.remove('index.html')
  versions.remove('.nojekyll')
  return "^".join(list(map(to_rexp, versions))) + include + "$"
