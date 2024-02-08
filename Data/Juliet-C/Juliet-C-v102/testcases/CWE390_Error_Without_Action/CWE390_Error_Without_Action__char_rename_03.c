/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE390_Error_Without_Action__char_rename_03.c
Label Definition File: CWE390_Error_Without_Action.string.label.xml
Template File: point-flaw-03.tmpl.c
*/
/*
 * @description
 * CWE: 390 Detection of Error Condition Without Action
 * Sinks: rename
 *    GoodSink: Check if rename() failed and handle errors properly
 *    BadSink : Check to see if rename() failed, but do nothing about it
 * Flow Variant: 03 Control flow: if(5==5) and if(5!=5)
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define RENAME rename
#else
# define RENAME rename
#endif

#define OLD_BAD_FILE_NAME "oldbadfilename.txt"
#define OLD_GOOD_FILE_NAME "oldgoodfilename.txt"

#ifndef OMITBAD

void CWE390_Error_Without_Action__char_rename_03_bad()
{
    if(5==5)
    {
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (RENAME(OLD_BAD_FILE_NAME, "newbadfilename.txt") != 0)
            {
                /* do nothing */
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FIX: check the return value and handle errors properly */
            if (RENAME(OLD_GOOD_FILE_NAME, "newgoodfilename.txt") != 0)
            {
                printLine("rename failed!");
                exit(1);
            }
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good1() uses if(5!=5) instead of if(5==5) */
static void good1()
{
    if(5!=5)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (RENAME(OLD_BAD_FILE_NAME, "newbadfilename.txt") != 0)
            {
                /* do nothing */
            }
        }
    }
    else
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (RENAME(OLD_GOOD_FILE_NAME, "newgoodfilename.txt") != 0)
            {
                printLine("rename failed!");
                exit(1);
            }
        }
    }
}

/* good2() reverses the bodies in the if statement */
static void good2()
{
    if(5==5)
    {
        {
            /* FIX: check the return value and handle errors properly */
            if (RENAME(OLD_GOOD_FILE_NAME, "newgoodfilename.txt") != 0)
            {
                printLine("rename failed!");
                exit(1);
            }
        }
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        {
            /* FLAW: check the return value, but do nothing if there is an error */
            if (RENAME(OLD_BAD_FILE_NAME, "newbadfilename.txt") != 0)
            {
                /* do nothing */
            }
        }
    }
}

void CWE390_Error_Without_Action__char_rename_03_good()
{
    good1();
    good2();
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
    CWE390_Error_Without_Action__char_rename_03_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE390_Error_Without_Action__char_rename_03_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
