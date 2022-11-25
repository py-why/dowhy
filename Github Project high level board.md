# Github Project - High Level Board

## Motivation
As the number of people contributing and taking interest in PyWhy is growing there is a desire to provide some structure to communicating what is being worked on and in what stage it is.

The Kanban board provided by Github is a grate way to run daily stand-ups within a work-stream discussing the resolutions of issues and bugs. It feels too detailed to serve the weekly Discord call.

## Suggested board approach
There are two possibilities to choose from for a high level "state of the project" review:
- Each column in the board represents a person
- Each column in the board represents an Epic or a Challenge

The PBIs ([Product Backlog Items](https://www.agile-academy.com/en/agile-dictionary/product-backlog-item-pbi/)) can start of as notes which is decided, can be turned into issues that to be monitored in Kanban style boards.

As the board is used, the incorporation of pull requests can be explored.

## The Devil is in the detail
The 2022 capability of managing a project alows for several view tabs. In the table view, it is possible to define a new field:
Name: Challenge
Type: Single select
Values: {
- DoWhy API V2,
- Causal Discovery,
- Causal Estimation,
- NetworkX,
- Documentation and Examples,
- GCM,
- bug fixing 
}