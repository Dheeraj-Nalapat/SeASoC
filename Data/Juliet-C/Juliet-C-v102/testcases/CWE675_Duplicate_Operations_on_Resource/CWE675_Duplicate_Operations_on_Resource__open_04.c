/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE675_Duplicate_Operations_on_Resource__open_04.c
Label Definition File: CWE675_Duplicate_Operations_on_Resource__open.label.xml
Template File: sources-sinks-04.tmpl.c
*/
/*
 * @description
 * CWE: 675 Duplicate Operations on Resource
 * BadSource:  Open and close a file using open() and close()
 * GoodSource: Open a file using open()
 * Sinks:
 *    GoodSink: Do nothing
 *    BadSink : Close the file
 * Flow Variant: 04 Control flow: if(static_const_t) and if(static_const_f)
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

/* The two variables below are declared "const", so a tool should
   be able to identify that reads of these will always return their
   initialized values. */
static const int static_const_t = 1; /* true */
static const int static_const_f = 0; /* false */

#ifndef OMITBAD

void CWE675_Duplicate_Operations_on_Resource__open_04_bad()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        /* POTENTIAL FLAW: Close the file in the source */
        CLOSE(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Open, but do not close the file in the source */
        data = OPEN("GoodSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    }
    if(static_const_t)
    {
        /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
        CLOSE(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Do nothing */
        /* FIX: Don't close the file in the sink */
        ; /* empty statement needed for some flow variants */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G1() - use badsource and goodsink by changing the second static_const_t to static_const_f */
static void goodB2G1()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        /* POTENTIAL FLAW: Close the file in the source */
        CLOSE(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Open, but do not close the file in the source */
        data = OPEN("GoodSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    }
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
        CLOSE(data);
    }
    else
    {
        /* Do nothing */
        /* FIX: Don't close the file in the sink */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodB2G2() - use badsource and goodsink by reversing the blocks in the second if */
static void goodB2G2()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        /* POTENTIAL FLAW: Close the file in the source */
        CLOSE(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Open, but do not close the file in the source */
        data = OPEN("GoodSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    }
    if(static_const_t)
    {
        /* Do nothing */
        /* FIX: Don't close the file in the sink */
        ; /* empty statement needed for some flow variants */
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
        CLOSE(data);
    }
}

/* goodG2B1() - use goodsource and badsink by changing the first static_const_t to static_const_f */
static void goodG2B1()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_f)
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        /* POTENTIAL FLAW: Close the file in the source */
        CLOSE(data);
    }
    else
    {
        /* FIX: Open, but do not close the file in the source */
        data = OPEN("GoodSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    }
    if(static_const_t)
    {
        /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
        CLOSE(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Do nothing */
        /* FIX: Don't close the file in the sink */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the first if */
static void goodG2B2()
{
    int data;
    data = -1; /* Initialize data */
    if(static_const_t)
    {
        /* FIX: Open, but do not close the file in the source */
        data = OPEN("GoodSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        data = OPEN("BadSource_open.txt", O_RDWR|O_CREAT, S_IREAD|S_IWRITE);
        /* POTENTIAL FLAW: Close the file in the source */
        CLOSE(data);
    }
    if(static_const_t)
    {
        /* POTENTIAL FLAW: Close the file in the sink (it may have been closed in the Source) */
        CLOSE(data);
    }
    else
    {
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* Do nothing */
        /* FIX: Don't close the file in the sink */
        ; /* empty statement needed for some flow variants */
    }
}

void CWE675_Duplicate_Operations_on_Resource__open_04_good()
{
    goodB2G1();
    goodB2G2();
    goodG2B1();
    goodG2B2();
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
    CWE675_Duplicate_Operations_on_Resource__open_04_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE675_Duplicate_Operations_on_Resource__open_04_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
