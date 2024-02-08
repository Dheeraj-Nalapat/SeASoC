/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__char_substring_ncmp_correct_pw_41.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-41.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: substring Provide a password that is a shortened version of the actual password
 * GoodSource: Provide a matching password
 * Sinks: ncmp_correct_pw
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use strncmp() to do password match, but use the length of the correct password
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD "Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ strlen(PASSWORD)

#ifndef OMITBAD

static void bad_sink(char * data)
{
    /* By using the strlen() of the correct password, you are able to create a partial comparison */
    /* For example if PASSWORD=PASSWORD1234, a user supplied password of PASSWORD123456 will allow access */
    /* POTENTIAL FLAW: Possibly partially compare the two passwords */
    if (strncmp(PASSWORD, data, strlen(PASSWORD)) == 0)
        printLine("Access granted");
    else
        printLine("Access denied!");
}

void CWE187_Partial_Comparison__char_substring_ncmp_correct_pw_41_bad()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    /* FLAW: Provide a shortened version of the actual password
     * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
     * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
     * and does not match the full PASSWORD string */
    data = "P";
    bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B_sink(char * data)
{
    /* By using the strlen() of the correct password, you are able to create a partial comparison */
    /* For example if PASSWORD=PASSWORD1234, a user supplied password of PASSWORD123456 will allow access */
    /* POTENTIAL FLAW: Possibly partially compare the two passwords */
    if (strncmp(PASSWORD, data, strlen(PASSWORD)) == 0)
        printLine("Access granted");
    else
        printLine("Access denied!");
}

static void goodG2B()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use the matching string */
    data = PASSWORD;
    goodG2B_sink(data);
}

/* goodB2G uses the BadSource with the GoodSink */
static void goodB2G_sink(char * data)
{
    /* Ideally, we would want to do a check to see if the passwords are of equal length */
    /* FIX: Compare the two passwords completely and correctly */
    if (strcmp(PASSWORD, data) == 0)
        printLine("Access granted");
    else
        printLine("Access denied!");
}

static void goodB2G()
{
    char * data;
    char data_buf[100] = "";
    data = data_buf;
    /* FLAW: Provide a shortened version of the actual password
     * NOTE: This must be a substring of PASSWORD starting with the first character in PASSWORD
     * i.e. other examples could be "Pa", "Pas", "Pass", etc. as long as it is shorter than MIN_PASSWORD_SZ
     * and does not match the full PASSWORD string */
    data = "P";
    goodB2G_sink(data);
}

void CWE187_Partial_Comparison__char_substring_ncmp_correct_pw_41_good()
{
    goodB2G();
    goodG2B();
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
    CWE187_Partial_Comparison__char_substring_ncmp_correct_pw_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__char_substring_ncmp_correct_pw_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
