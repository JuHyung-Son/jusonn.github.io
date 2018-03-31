# coding: utf-8

Gem::Specification.new do |spec|
  spec.name          = "Ju World"
  spec.version       = "0.2.3"
  spec.authors       = ["JuHyung Son"]
  spec.email         = ["sonju0427@gmail.com"]

  spec.summary       = %q{A blog of JuWorld}
  spec.homepage      = "https://jusonn.github.io"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r{^(assets|_layouts|_includes|_sass|LICENSE|README)}i) }

  spec.add_runtime_dependency "jekyll", "= 3.7.2"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.9.3"
  spec.add_runtime_dependency "jekyll-redirect-from", "~> 0.12.1"
  spec.add_runtime_dependency "jekyll-seo-tag", '~> 2.4', '>= 2.4.0'
  spec.add_runtime_dependency "jekyll-sitemap", '~> 1.2', '>= 1.2.0'

  spec.add_development_dependency "bundler", "~> 1.12"
  spec.add_development_dependency "rake", "~> 10.0"
end
