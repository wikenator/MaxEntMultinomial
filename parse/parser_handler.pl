#!/usr/bin/perl

# Author: Arnold Wikey
# Creation Date: January 22, 2013
# Description: Parser handler API for connecting other programming languages to the detexification Perl scripts

use Cwd qw(cwd);
use lib ('/home/arnold/git_repos/detexify');

use strict;
use warnings;
use Switch;
use IPC::Open2;
use PerlAPI qw(detex abstract);
#use PerlAPIold qw(verify expand_expr);

my %probInfo;

my $rawString = <STDIN>;
chomp($rawString);

#print "$rawString\n";
### Original String Info ######################################################
# $rawString = parser#@!$parser#@#problem#@!$problem
#my %probInfo = split /#@[#!]/, $rawString;
foreach (split('#@#', $rawString)) {
	my @values = split('#@!', $_);

	# handle answer type
	$probInfo{$values[0]} = $values[1];
}

switch ($probInfo{'parser'}) {
	case 'detex' {
		my @meta = split('@#@', $probInfo{'problem'});
		$probInfo{'problem'} = $meta[0];
		$probInfo{'problem'} = &detex($probInfo{'problem'}, $meta[1]);
	}
	case 'abstract' {
		my @meta = split('@#@', $probInfo{'problem'});
		$probInfo{'problem'} = $meta[0];
		$probInfo{'problem'} = &abstract($probInfo{'problem'}, $meta[1]);
	}
	case 'expand_expr' { $probInfo{'problem'} = &expand_expr($probInfo{'problem'}); }
#	case 'verify' {
#		my @answers = split('!@#', $probInfo{'problem'});

#		$probInfo{'problem'} = &verify($answers[0] . '#@#' . $answers[1]);
#	}
#	case 'unbalancedCharacter' {
#		my @problemInfo = split('!@#', $probInfo{'problem'});
#		my @charInfo = split(',', $problemInfo[1]);

#		$probInfo{'problem'} = &unbalancedCharacter($problemInfo[0], $charInfo[0], $charInfo[1], $charInfo[2]);
#	}
#	case 'simp' { $probInfo{'problem'} = &simp($probInfo{'problem'}); }
}

print $probInfo{'problem'};
exit();
