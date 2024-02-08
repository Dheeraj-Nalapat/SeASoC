/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__realloc_19.c
Label Definition File: CWE390_Error_Without_Action.label.xml
Template File: point-flaw-19.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: realloc
 *    GoodSink: Check to see if realloc() failed, and handle errors properly
 *    BadSink : Check to see if realloc() failed, but fail to handle errors
 * Flow Variant: 19 Control flow: Dead code after a return
 *
 * */

#include "std_testcase.h"

#ifndef OMITBAD

void CWE390_Error_Without_Action__realloc_19_bad()
{
    {
        char * data = NULL;
        data = (char *)realloc(data, 100);
        /* FLAW: Check to see if realloc() failed, but do nothing about it */
        if (data == NULL)
        {
            /* do nothing */
        }
        strcpy(data, "BadSinkBody");
        printLine(data);
        if (data != NULL)
        {
            free(data);
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        char * data = NULL;
        data = (char *)realloc(data, 100);
        /* FIX: Check to see if realloc() failed and handle errors properly */
        if (data == NULL)
        {
            printLine("realloc() failed");
            exit(1);
        }
        strcpy(data, "BadSinkBody");
        printLine(data);
        if (data != NULL)
        {
            free(data);
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() reverses the GoodSinkBody and the BadSinkBody so that the BadSinkBody never runs */
static void good1()
{
    {
        char * data = NULL;
        data = (char *)realloc(data, 100);
        /* FIX: Check to see if realloc() failed and handle errors properly */
        if (data == NULL)
        {
            printLine("realloc() failed");
            exit(1);
        }
        strcpy(data, "BadSinkBody");
        printLine(data);
        if (data != NULL)
        {
            free(data);
        }
    }
    return;
    /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        char * data = NULL;
        data = (char *)realloc(data, 100);
        /* FLAW: Check to see if realloc() failed, but do nothing about it */
        if (data == NULL)
        {
            /* do nothing */
        }
        strcpy(data, "BadSinkBody");
        printLine(data);
        if (data != NULL)
        {
            free(data);
        }
    }
}

void CWE390_Error_Without_Action__realloc_19_good()
{
    good1();
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
    CWE390_Error_Without_Action__realloc_19_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__realloc_19_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
