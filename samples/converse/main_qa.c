#include "converse_qa.h"


/** Entry point: grounded QA, attribution and grounded recombination. */
int main(int argc, char* argv[])
{
  converse_run_t run;
  int result = converse_setup(argc, argv, CONVERSE_ROLE_QA, &run);
  if (EXIT_SUCCESS == result && 0 != run.pending) result = converse_qa_run(&run);
  converse_release(&run);
  return result;
}
