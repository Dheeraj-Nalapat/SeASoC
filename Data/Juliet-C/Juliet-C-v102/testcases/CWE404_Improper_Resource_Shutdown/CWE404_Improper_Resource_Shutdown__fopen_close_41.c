/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE404_Improper_Resource_Shutdown__fopen_close_41.c
Label Definition File: CWE404_Improper_Resource_Shutdown.label.xml
Template File: source-sinks-41.tmpl.c
*/
/*
 * @description
 * CWE: 404 Improper Resource Shutdown or Release
 * BadSource: fopen Open a file using fopen()
 * Sinks: close
 *    GoodSink: Close the file using fclose()
 *    BadSink : Close the file using close()
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define CLOSE _close
#else
# define CLOSE close
#endif

#ifndef OMITBAD

static void bad_sink(FILE * data)
{
    if (data != NULL)
    {
        /* FLAW: Attempt to close the file using close() instead of fclose() */
        CLOSE((int)data);
    }
}

void CWE404_Improper_Resource_Shutdown__fopen_close_41_bad()
{
    FILE * data;
    /* Initialize data */
    data = NULL;
    data = fopen("BadSource_fopen.txt", "w+");
    bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

static void goodB2G_sink(FILE * data)
{
    if (data != NULL)
    {
        /* FIX: Close the file using fclose() */
        fclose(data);
    }
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G()
{
    FILE * data;
    /* Initialize data */
    data = NULL;
    data = fopen("BadSource_fopen.txt", "w+");
    goodB2G_sink(data);
}

void CWE404_Improper_Resource_Shutdown__fopen_close_41_good()
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
    CWE404_Improper_Resource_Shutdown__fopen_close_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE404_Improper_Resource_Shutdown__fopen_close_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
