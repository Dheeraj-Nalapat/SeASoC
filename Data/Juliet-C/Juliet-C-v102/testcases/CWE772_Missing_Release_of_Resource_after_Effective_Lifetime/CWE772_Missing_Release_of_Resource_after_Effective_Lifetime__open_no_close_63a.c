/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63a.c
Label Definition File: CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close.label.xml
Template File: source-sinks-63a.tmpl.c
*/
/*
 * @description
 * CWE: 772 Missing Release of Resource after Effective Lifetime
 * BadSource:  Open a file using open()
 * Sinks:
 *    GoodSink: Close the file using close()
 *    BadSink : Do not close file
 * Flow Variant: 63 Data flow: pointer to data passed from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define OPEN _open
# define CLOSE _close
#else
# define OPEN open
# define CLOSE close
#endif

#ifndef OMITBAD

/* bad function declaration */
void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63b_bad_sink(int * data_ptr);

void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63_bad()
{
    int data;
    /* Initialize data */
    data = -1;
    data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63b_bad_sink(&data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G uses the BadSource with the GoodSink */
void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63b_goodB2G_sink(int * data);

static void goodB2G()
{
    int data;
    /* Initialize data */
    data = -1;
    data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63b_goodB2G_sink(&data);
}

void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63_good()
{
    goodB2G();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__open_no_close_63_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
