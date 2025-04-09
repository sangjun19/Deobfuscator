	.file	"jebinshaju_system_software_Pass2_flatten.c"
	.text
	.globl	t1
	.bss
	.align 16
	.type	t1, @object
	.size	t1, 20
t1:
	.zero	20
	.globl	t3
	.align 8
	.type	t3, @object
	.size	t3, 10
t3:
	.zero	10
	.globl	label
	.align 8
	.type	label, @object
	.size	label, 10
label:
	.zero	10
	.globl	len
	.align 4
	.type	len, @object
	.size	len, 4
len:
	.zero	4
	.globl	operand
	.align 8
	.type	operand, @object
	.size	operand, 10
operand:
	.zero	10
	.globl	i
	.align 4
	.type	i, @object
	.size	i, 4
i:
	.zero	4
	.globl	s
	.align 4
	.type	s, @object
	.size	s, 4
s:
	.zero	4
	.globl	fp5
	.align 8
	.type	fp5, @object
	.size	fp5, 8
fp5:
	.zero	8
	.globl	start
	.align 4
	.type	start, @object
	.size	start, 4
start:
	.zero	4
	.globl	OT
	.align 32
	.type	OT, @object
	.size	OT, 600
OT:
	.zero	600
	.globl	fp2
	.align 8
	.type	fp2, @object
	.size	fp2, 8
fp2:
	.zero	8
	.globl	_TIG_IZ_NSyn_envp
	.align 8
	.type	_TIG_IZ_NSyn_envp, @object
	.size	_TIG_IZ_NSyn_envp, 8
_TIG_IZ_NSyn_envp:
	.zero	8
	.globl	t2
	.align 16
	.type	t2, @object
	.size	t2, 20
t2:
	.zero	20
	.globl	j
	.align 4
	.type	j, @object
	.size	j, 4
j:
	.zero	4
	.globl	flag
	.align 4
	.type	flag, @object
	.size	flag, 4
flag:
	.zero	4
	.globl	fp1
	.align 8
	.type	fp1, @object
	.size	fp1, 8
fp1:
	.zero	8
	.globl	fp4
	.align 8
	.type	fp4, @object
	.size	fp4, 8
fp4:
	.zero	8
	.globl	o
	.align 4
	.type	o, @object
	.size	o, 4
o:
	.zero	4
	.globl	opcode
	.align 8
	.type	opcode, @object
	.size	opcode, 10
opcode:
	.zero	10
	.globl	ST
	.align 32
	.type	ST, @object
	.size	ST, 480
ST:
	.zero	480
	.globl	locctr
	.align 4
	.type	locctr, @object
	.size	locctr, 4
locctr:
	.zero	4
	.globl	_TIG_IZ_NSyn_argc
	.align 4
	.type	_TIG_IZ_NSyn_argc, @object
	.size	_TIG_IZ_NSyn_argc, 4
_TIG_IZ_NSyn_argc:
	.zero	4
	.globl	opd
	.align 4
	.type	opd, @object
	.size	opd, 4
opd:
	.zero	4
	.globl	size
	.align 4
	.type	size, @object
	.size	size, 4
size:
	.zero	4
	.globl	fp3
	.align 8
	.type	fp3, @object
	.size	fp3, 8
fp3:
	.zero	8
	.globl	_TIG_IZ_NSyn_argv
	.align 8
	.type	_TIG_IZ_NSyn_argv, @object
	.size	_TIG_IZ_NSyn_argv, 8
_TIG_IZ_NSyn_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"%X"
.LC1:
	.string	"r"
.LC2:
	.string	"E:\\intermed.txt"
.LC3:
	.string	"E:\\optab.txt"
.LC4:
	.string	"w"
.LC5:
	.string	"E:\\objectprog.txt"
.LC6:
	.string	"%x%s%s%s"
.LC7:
	.string	"H^%s^%06X^%06X\n"
.LC8:
	.string	"T^%06X^"
.LC9:
	.string	"%02X%s\n"
.LC10:
	.string	"E^%06X\n"
.LC11:
	.string	"WORD"
.LC12:
	.string	"%s%s"
.LC13:
	.string	"END"
.LC14:
	.string	"%.*s"
.LC15:
	.string	"BYTE"
	.align 8
.LC16:
	.string	"Error: Symbol not found in SYMTAB"
.LC17:
	.string	"%d"
.LC18:
	.string	"%06X"
	.text
	.globl	pass2
	.type	pass2, @function
pass2:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$36, -72(%rbp)
.L61:
	cmpq	$44, -72(%rbp)
	ja	.L64
	movq	-72(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L36-.L4
	.long	.L35-.L4
	.long	.L34-.L4
	.long	.L64-.L4
	.long	.L33-.L4
	.long	.L32-.L4
	.long	.L31-.L4
	.long	.L65-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L29-.L4
	.long	.L28-.L4
	.long	.L27-.L4
	.long	.L26-.L4
	.long	.L25-.L4
	.long	.L64-.L4
	.long	.L24-.L4
	.long	.L23-.L4
	.long	.L22-.L4
	.long	.L21-.L4
	.long	.L20-.L4
	.long	.L19-.L4
	.long	.L64-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L64-.L4
	.long	.L16-.L4
	.long	.L64-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L64-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L64-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L23:
	movl	-116(%rbp), %edx
	leaq	-28(%rbp), %rax
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$17, -72(%rbp)
	jmp	.L37
.L17:
	cmpl	$0, -96(%rbp)
	je	.L38
	movq	$34, -72(%rbp)
	jmp	.L37
.L38:
	movq	$15, -72(%rbp)
	jmp	.L37
.L33:
	cmpl	$0, -100(%rbp)
	jne	.L40
	movq	$0, -72(%rbp)
	jmp	.L37
.L40:
	movq	$39, -72(%rbp)
	jmp	.L37
.L14:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp4(%rip)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp2(%rip)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp5(%rip)
	movq	fp4(%rip), %rax
	leaq	-120(%rbp), %rdx
	leaq	operand(%rip), %r9
	leaq	opcode(%rip), %r8
	leaq	label(%rip), %rcx
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movl	size(%rip), %ecx
	movl	start(%rip), %edx
	movq	fp5(%rip), %rax
	movl	%ecx, %r8d
	movl	%edx, %ecx
	leaq	label(%rip), %rdx
	leaq	.LC7(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-120(%rbp), %edx
	movq	fp5(%rip), %rax
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$38, -72(%rbp)
	jmp	.L37
.L26:
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$5, -72(%rbp)
	jmp	.L37
.L25:
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	shrq	%rax
	movq	%rax, %rsi
	movq	fp5(%rip), %rax
	leaq	-18(%rbp), %rdx
	movq	%rdx, %rcx
	movq	%rsi, %rdx
	leaq	.LC9(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	start(%rip), %edx
	movq	fp5(%rip), %rax
	leaq	.LC10(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	fp4(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	fp2(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	fp5(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$7, -72(%rbp)
	jmp	.L37
.L13:
	leaq	-38(%rbp), %rdx
	leaq	-18(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcat@PLT
	movq	$37, -72(%rbp)
	jmp	.L37
.L28:
	movl	$1, -112(%rbp)
	movq	$21, -72(%rbp)
	jmp	.L37
.L35:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -100(%rbp)
	movq	$4, -72(%rbp)
	jmp	.L37
.L18:
	movl	i(%rip), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	leaq	OT(%rip), %rdx
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -108(%rbp)
	movq	$27, -72(%rbp)
	jmp	.L37
.L20:
	cmpl	$0, -112(%rbp)
	je	.L42
	movq	$20, -72(%rbp)
	jmp	.L37
.L42:
	movq	$44, -72(%rbp)
	jmp	.L37
.L9:
	movq	$30, -72(%rbp)
	jmp	.L37
.L29:
	leaq	operand(%rip), %rax
	movq	%rax, %rdi
	call	search_SYMTAB
	movl	%eax, -116(%rbp)
	movq	$32, -72(%rbp)
	jmp	.L37
.L27:
	movl	$0, -112(%rbp)
	movl	$0, i(%rip)
	movq	$5, -72(%rbp)
	jmp	.L37
.L22:
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rax
	shrq	%rax
	movq	%rax, %rsi
	movq	fp5(%rip), %rax
	leaq	-18(%rbp), %rdx
	movq	%rdx, %rcx
	movq	%rsi, %rdx
	leaq	.LC9(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	-120(%rbp), %edx
	movq	fp5(%rip), %rax
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movb	$0, -18(%rbp)
	movq	$31, -72(%rbp)
	jmp	.L37
.L12:
	movl	-116(%rbp), %eax
	cmpl	$-1, %eax
	je	.L44
	movq	$18, -72(%rbp)
	jmp	.L37
.L44:
	movq	$33, -72(%rbp)
	jmp	.L37
.L24:
	movl	i(%rip), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	leaq	OT(%rip), %rdx
	addq	%rdx, %rax
	leaq	10(%rax), %rsi
	leaq	-28(%rbp), %rdx
	leaq	-38(%rbp), %rax
	movq	%rdx, %rcx
	movq	%rsi, %rdx
	leaq	.LC12(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$39, -72(%rbp)
	jmp	.L37
.L31:
	leaq	-28(%rbp), %rax
	movl	$808464432, (%rax)
	movb	$0, 4(%rax)
	movq	$17, -72(%rbp)
	jmp	.L37
.L16:
	cmpl	$0, -108(%rbp)
	jne	.L46
	movq	$12, -72(%rbp)
	jmp	.L37
.L46:
	movq	$14, -72(%rbp)
	jmp	.L37
.L7:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -96(%rbp)
	movq	$25, -72(%rbp)
	jmp	.L37
.L10:
	movzbl	opcode(%rip), %eax
	cmpb	$46, %al
	je	.L48
	movq	$13, -72(%rbp)
	jmp	.L37
.L48:
	movq	$37, -72(%rbp)
	jmp	.L37
.L19:
	leaq	operand(%rip), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	subl	$3, %eax
	movl	%eax, -92(%rbp)
	leaq	2+operand(%rip), %rcx
	movl	-92(%rbp), %edx
	leaq	-38(%rbp), %rax
	leaq	.LC14(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$39, -72(%rbp)
	jmp	.L37
.L3:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -104(%rbp)
	movq	$43, -72(%rbp)
	jmp	.L37
.L32:
	movl	i(%rip), %edx
	movl	o(%rip), %eax
	cmpl	%eax, %edx
	jg	.L50
	movq	$24, -72(%rbp)
	jmp	.L37
.L50:
	movq	$21, -72(%rbp)
	jmp	.L37
.L11:
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L8:
	movq	fp4(%rip), %rax
	leaq	-120(%rbp), %rdx
	leaq	operand(%rip), %r9
	leaq	opcode(%rip), %r8
	leaq	label(%rip), %rcx
	leaq	.LC6(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movq	$38, -72(%rbp)
	jmp	.L37
.L36:
	movzbl	opcode(%rip), %eax
	cmpb	$66, %al
	jne	.L52
	movq	$22, -72(%rbp)
	jmp	.L37
.L52:
	movq	$29, -72(%rbp)
	jmp	.L37
.L6:
	leaq	-18(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -88(%rbp)
	leaq	-38(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -80(%rbp)
	movq	$2, -72(%rbp)
	jmp	.L37
.L15:
	leaq	-116(%rbp), %rax
	movq	%rax, %rdx
	leaq	.LC17(%rip), %rax
	movq	%rax, %rsi
	leaq	operand(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_sscanf@PLT
	movl	-116(%rbp), %edx
	leaq	-38(%rbp), %rax
	leaq	.LC18(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	sprintf@PLT
	movq	$39, -72(%rbp)
	jmp	.L37
.L5:
	cmpl	$0, -104(%rbp)
	jne	.L55
	movq	$0, -72(%rbp)
	jmp	.L37
.L55:
	movq	$1, -72(%rbp)
	jmp	.L37
.L34:
	movq	-88(%rbp), %rdx
	movq	-80(%rbp), %rax
	addq	%rdx, %rax
	cmpq	$60, %rax
	jbe	.L57
	movq	$19, -72(%rbp)
	jmp	.L37
.L57:
	movq	$31, -72(%rbp)
	jmp	.L37
.L21:
	movzbl	operand(%rip), %eax
	cmpb	$45, %al
	je	.L59
	movq	$11, -72(%rbp)
	jmp	.L37
.L59:
	movq	$6, -72(%rbp)
	jmp	.L37
.L64:
	nop
.L37:
	jmp	.L61
.L65:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L63
	call	__stack_chk_fail@PLT
.L63:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	pass2, .-pass2
	.section	.rodata
.LC19:
	.string	"%x\n%x"
.LC20:
	.string	"%x\t%s\t%s\t%s\n"
.LC21:
	.string	"-"
.LC22:
	.string	"Error"
.LC23:
	.string	"RESW"
.LC24:
	.string	"\t%s\t%s\t%x\n"
.LC25:
	.string	"%s%s%s"
.LC26:
	.string	"\t%s\t%s\t%s\n"
.LC27:
	.string	"E:\\input.txt"
.LC28:
	.string	"E:\\symtab.txt"
.LC29:
	.string	"E:\\length.txt"
.LC30:
	.string	"%s%s%x"
.LC31:
	.string	"START"
.LC32:
	.string	"RESB"
.LC33:
	.string	"%s\t%x"
	.text
	.globl	main
	.type	main, @function
main:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	addq	$-128, %rsp
	movl	%edi, -100(%rbp)
	movq	%rsi, -112(%rbp)
	movq	%rdx, -120(%rbp)
	movb	$0, OT(%rip)
	movb	$0, 1+OT(%rip)
	movb	$0, 2+OT(%rip)
	movb	$0, 3+OT(%rip)
	movb	$0, 4+OT(%rip)
	movb	$0, 5+OT(%rip)
	movb	$0, 6+OT(%rip)
	movb	$0, 7+OT(%rip)
	movb	$0, 8+OT(%rip)
	movb	$0, 9+OT(%rip)
	movb	$0, 10+OT(%rip)
	movb	$0, 11+OT(%rip)
	movb	$0, 12+OT(%rip)
	movb	$0, 13+OT(%rip)
	movb	$0, 14+OT(%rip)
	movb	$0, 15+OT(%rip)
	movb	$0, 16+OT(%rip)
	movb	$0, 17+OT(%rip)
	movb	$0, 18+OT(%rip)
	movb	$0, 19+OT(%rip)
	movb	$0, 20+OT(%rip)
	movb	$0, 21+OT(%rip)
	movb	$0, 22+OT(%rip)
	movb	$0, 23+OT(%rip)
	movb	$0, 24+OT(%rip)
	movb	$0, 25+OT(%rip)
	movb	$0, 26+OT(%rip)
	movb	$0, 27+OT(%rip)
	movb	$0, 28+OT(%rip)
	movb	$0, 29+OT(%rip)
	movb	$0, 30+OT(%rip)
	movb	$0, 31+OT(%rip)
	movb	$0, 32+OT(%rip)
	movb	$0, 33+OT(%rip)
	movb	$0, 34+OT(%rip)
	movb	$0, 35+OT(%rip)
	movb	$0, 36+OT(%rip)
	movb	$0, 37+OT(%rip)
	movb	$0, 38+OT(%rip)
	movb	$0, 39+OT(%rip)
	movb	$0, 40+OT(%rip)
	movb	$0, 41+OT(%rip)
	movb	$0, 42+OT(%rip)
	movb	$0, 43+OT(%rip)
	movb	$0, 44+OT(%rip)
	movb	$0, 45+OT(%rip)
	movb	$0, 46+OT(%rip)
	movb	$0, 47+OT(%rip)
	movb	$0, 48+OT(%rip)
	movb	$0, 49+OT(%rip)
	movb	$0, 50+OT(%rip)
	movb	$0, 51+OT(%rip)
	movb	$0, 52+OT(%rip)
	movb	$0, 53+OT(%rip)
	movb	$0, 54+OT(%rip)
	movb	$0, 55+OT(%rip)
	movb	$0, 56+OT(%rip)
	movb	$0, 57+OT(%rip)
	movb	$0, 58+OT(%rip)
	movb	$0, 59+OT(%rip)
	movb	$0, 60+OT(%rip)
	movb	$0, 61+OT(%rip)
	movb	$0, 62+OT(%rip)
	movb	$0, 63+OT(%rip)
	movb	$0, 64+OT(%rip)
	movb	$0, 65+OT(%rip)
	movb	$0, 66+OT(%rip)
	movb	$0, 67+OT(%rip)
	movb	$0, 68+OT(%rip)
	movb	$0, 69+OT(%rip)
	movb	$0, 70+OT(%rip)
	movb	$0, 71+OT(%rip)
	movb	$0, 72+OT(%rip)
	movb	$0, 73+OT(%rip)
	movb	$0, 74+OT(%rip)
	movb	$0, 75+OT(%rip)
	movb	$0, 76+OT(%rip)
	movb	$0, 77+OT(%rip)
	movb	$0, 78+OT(%rip)
	movb	$0, 79+OT(%rip)
	movb	$0, 80+OT(%rip)
	movb	$0, 81+OT(%rip)
	movb	$0, 82+OT(%rip)
	movb	$0, 83+OT(%rip)
	movb	$0, 84+OT(%rip)
	movb	$0, 85+OT(%rip)
	movb	$0, 86+OT(%rip)
	movb	$0, 87+OT(%rip)
	movb	$0, 88+OT(%rip)
	movb	$0, 89+OT(%rip)
	movb	$0, 90+OT(%rip)
	movb	$0, 91+OT(%rip)
	movb	$0, 92+OT(%rip)
	movb	$0, 93+OT(%rip)
	movb	$0, 94+OT(%rip)
	movb	$0, 95+OT(%rip)
	movb	$0, 96+OT(%rip)
	movb	$0, 97+OT(%rip)
	movb	$0, 98+OT(%rip)
	movb	$0, 99+OT(%rip)
	movb	$0, 100+OT(%rip)
	movb	$0, 101+OT(%rip)
	movb	$0, 102+OT(%rip)
	movb	$0, 103+OT(%rip)
	movb	$0, 104+OT(%rip)
	movb	$0, 105+OT(%rip)
	movb	$0, 106+OT(%rip)
	movb	$0, 107+OT(%rip)
	movb	$0, 108+OT(%rip)
	movb	$0, 109+OT(%rip)
	movb	$0, 110+OT(%rip)
	movb	$0, 111+OT(%rip)
	movb	$0, 112+OT(%rip)
	movb	$0, 113+OT(%rip)
	movb	$0, 114+OT(%rip)
	movb	$0, 115+OT(%rip)
	movb	$0, 116+OT(%rip)
	movb	$0, 117+OT(%rip)
	movb	$0, 118+OT(%rip)
	movb	$0, 119+OT(%rip)
	movb	$0, 120+OT(%rip)
	movb	$0, 121+OT(%rip)
	movb	$0, 122+OT(%rip)
	movb	$0, 123+OT(%rip)
	movb	$0, 124+OT(%rip)
	movb	$0, 125+OT(%rip)
	movb	$0, 126+OT(%rip)
	movb	$0, 127+OT(%rip)
	movb	$0, 128+OT(%rip)
	movb	$0, 129+OT(%rip)
	movb	$0, 130+OT(%rip)
	movb	$0, 131+OT(%rip)
	movb	$0, 132+OT(%rip)
	movb	$0, 133+OT(%rip)
	movb	$0, 134+OT(%rip)
	movb	$0, 135+OT(%rip)
	movb	$0, 136+OT(%rip)
	movb	$0, 137+OT(%rip)
	movb	$0, 138+OT(%rip)
	movb	$0, 139+OT(%rip)
	movb	$0, 140+OT(%rip)
	movb	$0, 141+OT(%rip)
	movb	$0, 142+OT(%rip)
	movb	$0, 143+OT(%rip)
	movb	$0, 144+OT(%rip)
	movb	$0, 145+OT(%rip)
	movb	$0, 146+OT(%rip)
	movb	$0, 147+OT(%rip)
	movb	$0, 148+OT(%rip)
	movb	$0, 149+OT(%rip)
	movb	$0, 150+OT(%rip)
	movb	$0, 151+OT(%rip)
	movb	$0, 152+OT(%rip)
	movb	$0, 153+OT(%rip)
	movb	$0, 154+OT(%rip)
	movb	$0, 155+OT(%rip)
	movb	$0, 156+OT(%rip)
	movb	$0, 157+OT(%rip)
	movb	$0, 158+OT(%rip)
	movb	$0, 159+OT(%rip)
	movb	$0, 160+OT(%rip)
	movb	$0, 161+OT(%rip)
	movb	$0, 162+OT(%rip)
	movb	$0, 163+OT(%rip)
	movb	$0, 164+OT(%rip)
	movb	$0, 165+OT(%rip)
	movb	$0, 166+OT(%rip)
	movb	$0, 167+OT(%rip)
	movb	$0, 168+OT(%rip)
	movb	$0, 169+OT(%rip)
	movb	$0, 170+OT(%rip)
	movb	$0, 171+OT(%rip)
	movb	$0, 172+OT(%rip)
	movb	$0, 173+OT(%rip)
	movb	$0, 174+OT(%rip)
	movb	$0, 175+OT(%rip)
	movb	$0, 176+OT(%rip)
	movb	$0, 177+OT(%rip)
	movb	$0, 178+OT(%rip)
	movb	$0, 179+OT(%rip)
	movb	$0, 180+OT(%rip)
	movb	$0, 181+OT(%rip)
	movb	$0, 182+OT(%rip)
	movb	$0, 183+OT(%rip)
	movb	$0, 184+OT(%rip)
	movb	$0, 185+OT(%rip)
	movb	$0, 186+OT(%rip)
	movb	$0, 187+OT(%rip)
	movb	$0, 188+OT(%rip)
	movb	$0, 189+OT(%rip)
	movb	$0, 190+OT(%rip)
	movb	$0, 191+OT(%rip)
	movb	$0, 192+OT(%rip)
	movb	$0, 193+OT(%rip)
	movb	$0, 194+OT(%rip)
	movb	$0, 195+OT(%rip)
	movb	$0, 196+OT(%rip)
	movb	$0, 197+OT(%rip)
	movb	$0, 198+OT(%rip)
	movb	$0, 199+OT(%rip)
	movb	$0, 200+OT(%rip)
	movb	$0, 201+OT(%rip)
	movb	$0, 202+OT(%rip)
	movb	$0, 203+OT(%rip)
	movb	$0, 204+OT(%rip)
	movb	$0, 205+OT(%rip)
	movb	$0, 206+OT(%rip)
	movb	$0, 207+OT(%rip)
	movb	$0, 208+OT(%rip)
	movb	$0, 209+OT(%rip)
	movb	$0, 210+OT(%rip)
	movb	$0, 211+OT(%rip)
	movb	$0, 212+OT(%rip)
	movb	$0, 213+OT(%rip)
	movb	$0, 214+OT(%rip)
	movb	$0, 215+OT(%rip)
	movb	$0, 216+OT(%rip)
	movb	$0, 217+OT(%rip)
	movb	$0, 218+OT(%rip)
	movb	$0, 219+OT(%rip)
	movb	$0, 220+OT(%rip)
	movb	$0, 221+OT(%rip)
	movb	$0, 222+OT(%rip)
	movb	$0, 223+OT(%rip)
	movb	$0, 224+OT(%rip)
	movb	$0, 225+OT(%rip)
	movb	$0, 226+OT(%rip)
	movb	$0, 227+OT(%rip)
	movb	$0, 228+OT(%rip)
	movb	$0, 229+OT(%rip)
	movb	$0, 230+OT(%rip)
	movb	$0, 231+OT(%rip)
	movb	$0, 232+OT(%rip)
	movb	$0, 233+OT(%rip)
	movb	$0, 234+OT(%rip)
	movb	$0, 235+OT(%rip)
	movb	$0, 236+OT(%rip)
	movb	$0, 237+OT(%rip)
	movb	$0, 238+OT(%rip)
	movb	$0, 239+OT(%rip)
	movb	$0, 240+OT(%rip)
	movb	$0, 241+OT(%rip)
	movb	$0, 242+OT(%rip)
	movb	$0, 243+OT(%rip)
	movb	$0, 244+OT(%rip)
	movb	$0, 245+OT(%rip)
	movb	$0, 246+OT(%rip)
	movb	$0, 247+OT(%rip)
	movb	$0, 248+OT(%rip)
	movb	$0, 249+OT(%rip)
	movb	$0, 250+OT(%rip)
	movb	$0, 251+OT(%rip)
	movb	$0, 252+OT(%rip)
	movb	$0, 253+OT(%rip)
	movb	$0, 254+OT(%rip)
	movb	$0, 255+OT(%rip)
	movb	$0, 256+OT(%rip)
	movb	$0, 257+OT(%rip)
	movb	$0, 258+OT(%rip)
	movb	$0, 259+OT(%rip)
	movb	$0, 260+OT(%rip)
	movb	$0, 261+OT(%rip)
	movb	$0, 262+OT(%rip)
	movb	$0, 263+OT(%rip)
	movb	$0, 264+OT(%rip)
	movb	$0, 265+OT(%rip)
	movb	$0, 266+OT(%rip)
	movb	$0, 267+OT(%rip)
	movb	$0, 268+OT(%rip)
	movb	$0, 269+OT(%rip)
	movb	$0, 270+OT(%rip)
	movb	$0, 271+OT(%rip)
	movb	$0, 272+OT(%rip)
	movb	$0, 273+OT(%rip)
	movb	$0, 274+OT(%rip)
	movb	$0, 275+OT(%rip)
	movb	$0, 276+OT(%rip)
	movb	$0, 277+OT(%rip)
	movb	$0, 278+OT(%rip)
	movb	$0, 279+OT(%rip)
	movb	$0, 280+OT(%rip)
	movb	$0, 281+OT(%rip)
	movb	$0, 282+OT(%rip)
	movb	$0, 283+OT(%rip)
	movb	$0, 284+OT(%rip)
	movb	$0, 285+OT(%rip)
	movb	$0, 286+OT(%rip)
	movb	$0, 287+OT(%rip)
	movb	$0, 288+OT(%rip)
	movb	$0, 289+OT(%rip)
	movb	$0, 290+OT(%rip)
	movb	$0, 291+OT(%rip)
	movb	$0, 292+OT(%rip)
	movb	$0, 293+OT(%rip)
	movb	$0, 294+OT(%rip)
	movb	$0, 295+OT(%rip)
	movb	$0, 296+OT(%rip)
	movb	$0, 297+OT(%rip)
	movb	$0, 298+OT(%rip)
	movb	$0, 299+OT(%rip)
	movb	$0, 300+OT(%rip)
	movb	$0, 301+OT(%rip)
	movb	$0, 302+OT(%rip)
	movb	$0, 303+OT(%rip)
	movb	$0, 304+OT(%rip)
	movb	$0, 305+OT(%rip)
	movb	$0, 306+OT(%rip)
	movb	$0, 307+OT(%rip)
	movb	$0, 308+OT(%rip)
	movb	$0, 309+OT(%rip)
	movb	$0, 310+OT(%rip)
	movb	$0, 311+OT(%rip)
	movb	$0, 312+OT(%rip)
	movb	$0, 313+OT(%rip)
	movb	$0, 314+OT(%rip)
	movb	$0, 315+OT(%rip)
	movb	$0, 316+OT(%rip)
	movb	$0, 317+OT(%rip)
	movb	$0, 318+OT(%rip)
	movb	$0, 319+OT(%rip)
	movb	$0, 320+OT(%rip)
	movb	$0, 321+OT(%rip)
	movb	$0, 322+OT(%rip)
	movb	$0, 323+OT(%rip)
	movb	$0, 324+OT(%rip)
	movb	$0, 325+OT(%rip)
	movb	$0, 326+OT(%rip)
	movb	$0, 327+OT(%rip)
	movb	$0, 328+OT(%rip)
	movb	$0, 329+OT(%rip)
	movb	$0, 330+OT(%rip)
	movb	$0, 331+OT(%rip)
	movb	$0, 332+OT(%rip)
	movb	$0, 333+OT(%rip)
	movb	$0, 334+OT(%rip)
	movb	$0, 335+OT(%rip)
	movb	$0, 336+OT(%rip)
	movb	$0, 337+OT(%rip)
	movb	$0, 338+OT(%rip)
	movb	$0, 339+OT(%rip)
	movb	$0, 340+OT(%rip)
	movb	$0, 341+OT(%rip)
	movb	$0, 342+OT(%rip)
	movb	$0, 343+OT(%rip)
	movb	$0, 344+OT(%rip)
	movb	$0, 345+OT(%rip)
	movb	$0, 346+OT(%rip)
	movb	$0, 347+OT(%rip)
	movb	$0, 348+OT(%rip)
	movb	$0, 349+OT(%rip)
	movb	$0, 350+OT(%rip)
	movb	$0, 351+OT(%rip)
	movb	$0, 352+OT(%rip)
	movb	$0, 353+OT(%rip)
	movb	$0, 354+OT(%rip)
	movb	$0, 355+OT(%rip)
	movb	$0, 356+OT(%rip)
	movb	$0, 357+OT(%rip)
	movb	$0, 358+OT(%rip)
	movb	$0, 359+OT(%rip)
	movb	$0, 360+OT(%rip)
	movb	$0, 361+OT(%rip)
	movb	$0, 362+OT(%rip)
	movb	$0, 363+OT(%rip)
	movb	$0, 364+OT(%rip)
	movb	$0, 365+OT(%rip)
	movb	$0, 366+OT(%rip)
	movb	$0, 367+OT(%rip)
	movb	$0, 368+OT(%rip)
	movb	$0, 369+OT(%rip)
	movb	$0, 370+OT(%rip)
	movb	$0, 371+OT(%rip)
	movb	$0, 372+OT(%rip)
	movb	$0, 373+OT(%rip)
	movb	$0, 374+OT(%rip)
	movb	$0, 375+OT(%rip)
	movb	$0, 376+OT(%rip)
	movb	$0, 377+OT(%rip)
	movb	$0, 378+OT(%rip)
	movb	$0, 379+OT(%rip)
	movb	$0, 380+OT(%rip)
	movb	$0, 381+OT(%rip)
	movb	$0, 382+OT(%rip)
	movb	$0, 383+OT(%rip)
	movb	$0, 384+OT(%rip)
	movb	$0, 385+OT(%rip)
	movb	$0, 386+OT(%rip)
	movb	$0, 387+OT(%rip)
	movb	$0, 388+OT(%rip)
	movb	$0, 389+OT(%rip)
	movb	$0, 390+OT(%rip)
	movb	$0, 391+OT(%rip)
	movb	$0, 392+OT(%rip)
	movb	$0, 393+OT(%rip)
	movb	$0, 394+OT(%rip)
	movb	$0, 395+OT(%rip)
	movb	$0, 396+OT(%rip)
	movb	$0, 397+OT(%rip)
	movb	$0, 398+OT(%rip)
	movb	$0, 399+OT(%rip)
	movb	$0, 400+OT(%rip)
	movb	$0, 401+OT(%rip)
	movb	$0, 402+OT(%rip)
	movb	$0, 403+OT(%rip)
	movb	$0, 404+OT(%rip)
	movb	$0, 405+OT(%rip)
	movb	$0, 406+OT(%rip)
	movb	$0, 407+OT(%rip)
	movb	$0, 408+OT(%rip)
	movb	$0, 409+OT(%rip)
	movb	$0, 410+OT(%rip)
	movb	$0, 411+OT(%rip)
	movb	$0, 412+OT(%rip)
	movb	$0, 413+OT(%rip)
	movb	$0, 414+OT(%rip)
	movb	$0, 415+OT(%rip)
	movb	$0, 416+OT(%rip)
	movb	$0, 417+OT(%rip)
	movb	$0, 418+OT(%rip)
	movb	$0, 419+OT(%rip)
	movb	$0, 420+OT(%rip)
	movb	$0, 421+OT(%rip)
	movb	$0, 422+OT(%rip)
	movb	$0, 423+OT(%rip)
	movb	$0, 424+OT(%rip)
	movb	$0, 425+OT(%rip)
	movb	$0, 426+OT(%rip)
	movb	$0, 427+OT(%rip)
	movb	$0, 428+OT(%rip)
	movb	$0, 429+OT(%rip)
	movb	$0, 430+OT(%rip)
	movb	$0, 431+OT(%rip)
	movb	$0, 432+OT(%rip)
	movb	$0, 433+OT(%rip)
	movb	$0, 434+OT(%rip)
	movb	$0, 435+OT(%rip)
	movb	$0, 436+OT(%rip)
	movb	$0, 437+OT(%rip)
	movb	$0, 438+OT(%rip)
	movb	$0, 439+OT(%rip)
	movb	$0, 440+OT(%rip)
	movb	$0, 441+OT(%rip)
	movb	$0, 442+OT(%rip)
	movb	$0, 443+OT(%rip)
	movb	$0, 444+OT(%rip)
	movb	$0, 445+OT(%rip)
	movb	$0, 446+OT(%rip)
	movb	$0, 447+OT(%rip)
	movb	$0, 448+OT(%rip)
	movb	$0, 449+OT(%rip)
	movb	$0, 450+OT(%rip)
	movb	$0, 451+OT(%rip)
	movb	$0, 452+OT(%rip)
	movb	$0, 453+OT(%rip)
	movb	$0, 454+OT(%rip)
	movb	$0, 455+OT(%rip)
	movb	$0, 456+OT(%rip)
	movb	$0, 457+OT(%rip)
	movb	$0, 458+OT(%rip)
	movb	$0, 459+OT(%rip)
	movb	$0, 460+OT(%rip)
	movb	$0, 461+OT(%rip)
	movb	$0, 462+OT(%rip)
	movb	$0, 463+OT(%rip)
	movb	$0, 464+OT(%rip)
	movb	$0, 465+OT(%rip)
	movb	$0, 466+OT(%rip)
	movb	$0, 467+OT(%rip)
	movb	$0, 468+OT(%rip)
	movb	$0, 469+OT(%rip)
	movb	$0, 470+OT(%rip)
	movb	$0, 471+OT(%rip)
	movb	$0, 472+OT(%rip)
	movb	$0, 473+OT(%rip)
	movb	$0, 474+OT(%rip)
	movb	$0, 475+OT(%rip)
	movb	$0, 476+OT(%rip)
	movb	$0, 477+OT(%rip)
	movb	$0, 478+OT(%rip)
	movb	$0, 479+OT(%rip)
	movb	$0, 480+OT(%rip)
	movb	$0, 481+OT(%rip)
	movb	$0, 482+OT(%rip)
	movb	$0, 483+OT(%rip)
	movb	$0, 484+OT(%rip)
	movb	$0, 485+OT(%rip)
	movb	$0, 486+OT(%rip)
	movb	$0, 487+OT(%rip)
	movb	$0, 488+OT(%rip)
	movb	$0, 489+OT(%rip)
	movb	$0, 490+OT(%rip)
	movb	$0, 491+OT(%rip)
	movb	$0, 492+OT(%rip)
	movb	$0, 493+OT(%rip)
	movb	$0, 494+OT(%rip)
	movb	$0, 495+OT(%rip)
	movb	$0, 496+OT(%rip)
	movb	$0, 497+OT(%rip)
	movb	$0, 498+OT(%rip)
	movb	$0, 499+OT(%rip)
	movb	$0, 500+OT(%rip)
	movb	$0, 501+OT(%rip)
	movb	$0, 502+OT(%rip)
	movb	$0, 503+OT(%rip)
	movb	$0, 504+OT(%rip)
	movb	$0, 505+OT(%rip)
	movb	$0, 506+OT(%rip)
	movb	$0, 507+OT(%rip)
	movb	$0, 508+OT(%rip)
	movb	$0, 509+OT(%rip)
	movb	$0, 510+OT(%rip)
	movb	$0, 511+OT(%rip)
	movb	$0, 512+OT(%rip)
	movb	$0, 513+OT(%rip)
	movb	$0, 514+OT(%rip)
	movb	$0, 515+OT(%rip)
	movb	$0, 516+OT(%rip)
	movb	$0, 517+OT(%rip)
	movb	$0, 518+OT(%rip)
	movb	$0, 519+OT(%rip)
	movb	$0, 520+OT(%rip)
	movb	$0, 521+OT(%rip)
	movb	$0, 522+OT(%rip)
	movb	$0, 523+OT(%rip)
	movb	$0, 524+OT(%rip)
	movb	$0, 525+OT(%rip)
	movb	$0, 526+OT(%rip)
	movb	$0, 527+OT(%rip)
	movb	$0, 528+OT(%rip)
	movb	$0, 529+OT(%rip)
	movb	$0, 530+OT(%rip)
	movb	$0, 531+OT(%rip)
	movb	$0, 532+OT(%rip)
	movb	$0, 533+OT(%rip)
	movb	$0, 534+OT(%rip)
	movb	$0, 535+OT(%rip)
	movb	$0, 536+OT(%rip)
	movb	$0, 537+OT(%rip)
	movb	$0, 538+OT(%rip)
	movb	$0, 539+OT(%rip)
	movb	$0, 540+OT(%rip)
	movb	$0, 541+OT(%rip)
	movb	$0, 542+OT(%rip)
	movb	$0, 543+OT(%rip)
	movb	$0, 544+OT(%rip)
	movb	$0, 545+OT(%rip)
	movb	$0, 546+OT(%rip)
	movb	$0, 547+OT(%rip)
	movb	$0, 548+OT(%rip)
	movb	$0, 549+OT(%rip)
	movb	$0, 550+OT(%rip)
	movb	$0, 551+OT(%rip)
	movb	$0, 552+OT(%rip)
	movb	$0, 553+OT(%rip)
	movb	$0, 554+OT(%rip)
	movb	$0, 555+OT(%rip)
	movb	$0, 556+OT(%rip)
	movb	$0, 557+OT(%rip)
	movb	$0, 558+OT(%rip)
	movb	$0, 559+OT(%rip)
	movb	$0, 560+OT(%rip)
	movb	$0, 561+OT(%rip)
	movb	$0, 562+OT(%rip)
	movb	$0, 563+OT(%rip)
	movb	$0, 564+OT(%rip)
	movb	$0, 565+OT(%rip)
	movb	$0, 566+OT(%rip)
	movb	$0, 567+OT(%rip)
	movb	$0, 568+OT(%rip)
	movb	$0, 569+OT(%rip)
	movb	$0, 570+OT(%rip)
	movb	$0, 571+OT(%rip)
	movb	$0, 572+OT(%rip)
	movb	$0, 573+OT(%rip)
	movb	$0, 574+OT(%rip)
	movb	$0, 575+OT(%rip)
	movb	$0, 576+OT(%rip)
	movb	$0, 577+OT(%rip)
	movb	$0, 578+OT(%rip)
	movb	$0, 579+OT(%rip)
	movb	$0, 580+OT(%rip)
	movb	$0, 581+OT(%rip)
	movb	$0, 582+OT(%rip)
	movb	$0, 583+OT(%rip)
	movb	$0, 584+OT(%rip)
	movb	$0, 585+OT(%rip)
	movb	$0, 586+OT(%rip)
	movb	$0, 587+OT(%rip)
	movb	$0, 588+OT(%rip)
	movb	$0, 589+OT(%rip)
	movb	$0, 590+OT(%rip)
	movb	$0, 591+OT(%rip)
	movb	$0, 592+OT(%rip)
	movb	$0, 593+OT(%rip)
	movb	$0, 594+OT(%rip)
	movb	$0, 595+OT(%rip)
	movb	$0, 596+OT(%rip)
	movb	$0, 597+OT(%rip)
	movb	$0, 598+OT(%rip)
	movb	$0, 599+OT(%rip)
	nop
.L67:
	movb	$0, ST(%rip)
	movb	$0, 1+ST(%rip)
	movb	$0, 2+ST(%rip)
	movb	$0, 3+ST(%rip)
	movb	$0, 4+ST(%rip)
	movb	$0, 5+ST(%rip)
	movb	$0, 6+ST(%rip)
	movb	$0, 7+ST(%rip)
	movb	$0, 8+ST(%rip)
	movb	$0, 9+ST(%rip)
	movl	$0, 12+ST(%rip)
	movb	$0, 16+ST(%rip)
	movb	$0, 17+ST(%rip)
	movb	$0, 18+ST(%rip)
	movb	$0, 19+ST(%rip)
	movb	$0, 20+ST(%rip)
	movb	$0, 21+ST(%rip)
	movb	$0, 22+ST(%rip)
	movb	$0, 23+ST(%rip)
	movb	$0, 24+ST(%rip)
	movb	$0, 25+ST(%rip)
	movl	$0, 28+ST(%rip)
	movb	$0, 32+ST(%rip)
	movb	$0, 33+ST(%rip)
	movb	$0, 34+ST(%rip)
	movb	$0, 35+ST(%rip)
	movb	$0, 36+ST(%rip)
	movb	$0, 37+ST(%rip)
	movb	$0, 38+ST(%rip)
	movb	$0, 39+ST(%rip)
	movb	$0, 40+ST(%rip)
	movb	$0, 41+ST(%rip)
	movl	$0, 44+ST(%rip)
	movb	$0, 48+ST(%rip)
	movb	$0, 49+ST(%rip)
	movb	$0, 50+ST(%rip)
	movb	$0, 51+ST(%rip)
	movb	$0, 52+ST(%rip)
	movb	$0, 53+ST(%rip)
	movb	$0, 54+ST(%rip)
	movb	$0, 55+ST(%rip)
	movb	$0, 56+ST(%rip)
	movb	$0, 57+ST(%rip)
	movl	$0, 60+ST(%rip)
	movb	$0, 64+ST(%rip)
	movb	$0, 65+ST(%rip)
	movb	$0, 66+ST(%rip)
	movb	$0, 67+ST(%rip)
	movb	$0, 68+ST(%rip)
	movb	$0, 69+ST(%rip)
	movb	$0, 70+ST(%rip)
	movb	$0, 71+ST(%rip)
	movb	$0, 72+ST(%rip)
	movb	$0, 73+ST(%rip)
	movl	$0, 76+ST(%rip)
	movb	$0, 80+ST(%rip)
	movb	$0, 81+ST(%rip)
	movb	$0, 82+ST(%rip)
	movb	$0, 83+ST(%rip)
	movb	$0, 84+ST(%rip)
	movb	$0, 85+ST(%rip)
	movb	$0, 86+ST(%rip)
	movb	$0, 87+ST(%rip)
	movb	$0, 88+ST(%rip)
	movb	$0, 89+ST(%rip)
	movl	$0, 92+ST(%rip)
	movb	$0, 96+ST(%rip)
	movb	$0, 97+ST(%rip)
	movb	$0, 98+ST(%rip)
	movb	$0, 99+ST(%rip)
	movb	$0, 100+ST(%rip)
	movb	$0, 101+ST(%rip)
	movb	$0, 102+ST(%rip)
	movb	$0, 103+ST(%rip)
	movb	$0, 104+ST(%rip)
	movb	$0, 105+ST(%rip)
	movl	$0, 108+ST(%rip)
	movb	$0, 112+ST(%rip)
	movb	$0, 113+ST(%rip)
	movb	$0, 114+ST(%rip)
	movb	$0, 115+ST(%rip)
	movb	$0, 116+ST(%rip)
	movb	$0, 117+ST(%rip)
	movb	$0, 118+ST(%rip)
	movb	$0, 119+ST(%rip)
	movb	$0, 120+ST(%rip)
	movb	$0, 121+ST(%rip)
	movl	$0, 124+ST(%rip)
	movb	$0, 128+ST(%rip)
	movb	$0, 129+ST(%rip)
	movb	$0, 130+ST(%rip)
	movb	$0, 131+ST(%rip)
	movb	$0, 132+ST(%rip)
	movb	$0, 133+ST(%rip)
	movb	$0, 134+ST(%rip)
	movb	$0, 135+ST(%rip)
	movb	$0, 136+ST(%rip)
	movb	$0, 137+ST(%rip)
	movl	$0, 140+ST(%rip)
	movb	$0, 144+ST(%rip)
	movb	$0, 145+ST(%rip)
	movb	$0, 146+ST(%rip)
	movb	$0, 147+ST(%rip)
	movb	$0, 148+ST(%rip)
	movb	$0, 149+ST(%rip)
	movb	$0, 150+ST(%rip)
	movb	$0, 151+ST(%rip)
	movb	$0, 152+ST(%rip)
	movb	$0, 153+ST(%rip)
	movl	$0, 156+ST(%rip)
	movb	$0, 160+ST(%rip)
	movb	$0, 161+ST(%rip)
	movb	$0, 162+ST(%rip)
	movb	$0, 163+ST(%rip)
	movb	$0, 164+ST(%rip)
	movb	$0, 165+ST(%rip)
	movb	$0, 166+ST(%rip)
	movb	$0, 167+ST(%rip)
	movb	$0, 168+ST(%rip)
	movb	$0, 169+ST(%rip)
	movl	$0, 172+ST(%rip)
	movb	$0, 176+ST(%rip)
	movb	$0, 177+ST(%rip)
	movb	$0, 178+ST(%rip)
	movb	$0, 179+ST(%rip)
	movb	$0, 180+ST(%rip)
	movb	$0, 181+ST(%rip)
	movb	$0, 182+ST(%rip)
	movb	$0, 183+ST(%rip)
	movb	$0, 184+ST(%rip)
	movb	$0, 185+ST(%rip)
	movl	$0, 188+ST(%rip)
	movb	$0, 192+ST(%rip)
	movb	$0, 193+ST(%rip)
	movb	$0, 194+ST(%rip)
	movb	$0, 195+ST(%rip)
	movb	$0, 196+ST(%rip)
	movb	$0, 197+ST(%rip)
	movb	$0, 198+ST(%rip)
	movb	$0, 199+ST(%rip)
	movb	$0, 200+ST(%rip)
	movb	$0, 201+ST(%rip)
	movl	$0, 204+ST(%rip)
	movb	$0, 208+ST(%rip)
	movb	$0, 209+ST(%rip)
	movb	$0, 210+ST(%rip)
	movb	$0, 211+ST(%rip)
	movb	$0, 212+ST(%rip)
	movb	$0, 213+ST(%rip)
	movb	$0, 214+ST(%rip)
	movb	$0, 215+ST(%rip)
	movb	$0, 216+ST(%rip)
	movb	$0, 217+ST(%rip)
	movl	$0, 220+ST(%rip)
	movb	$0, 224+ST(%rip)
	movb	$0, 225+ST(%rip)
	movb	$0, 226+ST(%rip)
	movb	$0, 227+ST(%rip)
	movb	$0, 228+ST(%rip)
	movb	$0, 229+ST(%rip)
	movb	$0, 230+ST(%rip)
	movb	$0, 231+ST(%rip)
	movb	$0, 232+ST(%rip)
	movb	$0, 233+ST(%rip)
	movl	$0, 236+ST(%rip)
	movb	$0, 240+ST(%rip)
	movb	$0, 241+ST(%rip)
	movb	$0, 242+ST(%rip)
	movb	$0, 243+ST(%rip)
	movb	$0, 244+ST(%rip)
	movb	$0, 245+ST(%rip)
	movb	$0, 246+ST(%rip)
	movb	$0, 247+ST(%rip)
	movb	$0, 248+ST(%rip)
	movb	$0, 249+ST(%rip)
	movl	$0, 252+ST(%rip)
	movb	$0, 256+ST(%rip)
	movb	$0, 257+ST(%rip)
	movb	$0, 258+ST(%rip)
	movb	$0, 259+ST(%rip)
	movb	$0, 260+ST(%rip)
	movb	$0, 261+ST(%rip)
	movb	$0, 262+ST(%rip)
	movb	$0, 263+ST(%rip)
	movb	$0, 264+ST(%rip)
	movb	$0, 265+ST(%rip)
	movl	$0, 268+ST(%rip)
	movb	$0, 272+ST(%rip)
	movb	$0, 273+ST(%rip)
	movb	$0, 274+ST(%rip)
	movb	$0, 275+ST(%rip)
	movb	$0, 276+ST(%rip)
	movb	$0, 277+ST(%rip)
	movb	$0, 278+ST(%rip)
	movb	$0, 279+ST(%rip)
	movb	$0, 280+ST(%rip)
	movb	$0, 281+ST(%rip)
	movl	$0, 284+ST(%rip)
	movb	$0, 288+ST(%rip)
	movb	$0, 289+ST(%rip)
	movb	$0, 290+ST(%rip)
	movb	$0, 291+ST(%rip)
	movb	$0, 292+ST(%rip)
	movb	$0, 293+ST(%rip)
	movb	$0, 294+ST(%rip)
	movb	$0, 295+ST(%rip)
	movb	$0, 296+ST(%rip)
	movb	$0, 297+ST(%rip)
	movl	$0, 300+ST(%rip)
	movb	$0, 304+ST(%rip)
	movb	$0, 305+ST(%rip)
	movb	$0, 306+ST(%rip)
	movb	$0, 307+ST(%rip)
	movb	$0, 308+ST(%rip)
	movb	$0, 309+ST(%rip)
	movb	$0, 310+ST(%rip)
	movb	$0, 311+ST(%rip)
	movb	$0, 312+ST(%rip)
	movb	$0, 313+ST(%rip)
	movl	$0, 316+ST(%rip)
	movb	$0, 320+ST(%rip)
	movb	$0, 321+ST(%rip)
	movb	$0, 322+ST(%rip)
	movb	$0, 323+ST(%rip)
	movb	$0, 324+ST(%rip)
	movb	$0, 325+ST(%rip)
	movb	$0, 326+ST(%rip)
	movb	$0, 327+ST(%rip)
	movb	$0, 328+ST(%rip)
	movb	$0, 329+ST(%rip)
	movl	$0, 332+ST(%rip)
	movb	$0, 336+ST(%rip)
	movb	$0, 337+ST(%rip)
	movb	$0, 338+ST(%rip)
	movb	$0, 339+ST(%rip)
	movb	$0, 340+ST(%rip)
	movb	$0, 341+ST(%rip)
	movb	$0, 342+ST(%rip)
	movb	$0, 343+ST(%rip)
	movb	$0, 344+ST(%rip)
	movb	$0, 345+ST(%rip)
	movl	$0, 348+ST(%rip)
	movb	$0, 352+ST(%rip)
	movb	$0, 353+ST(%rip)
	movb	$0, 354+ST(%rip)
	movb	$0, 355+ST(%rip)
	movb	$0, 356+ST(%rip)
	movb	$0, 357+ST(%rip)
	movb	$0, 358+ST(%rip)
	movb	$0, 359+ST(%rip)
	movb	$0, 360+ST(%rip)
	movb	$0, 361+ST(%rip)
	movl	$0, 364+ST(%rip)
	movb	$0, 368+ST(%rip)
	movb	$0, 369+ST(%rip)
	movb	$0, 370+ST(%rip)
	movb	$0, 371+ST(%rip)
	movb	$0, 372+ST(%rip)
	movb	$0, 373+ST(%rip)
	movb	$0, 374+ST(%rip)
	movb	$0, 375+ST(%rip)
	movb	$0, 376+ST(%rip)
	movb	$0, 377+ST(%rip)
	movl	$0, 380+ST(%rip)
	movb	$0, 384+ST(%rip)
	movb	$0, 385+ST(%rip)
	movb	$0, 386+ST(%rip)
	movb	$0, 387+ST(%rip)
	movb	$0, 388+ST(%rip)
	movb	$0, 389+ST(%rip)
	movb	$0, 390+ST(%rip)
	movb	$0, 391+ST(%rip)
	movb	$0, 392+ST(%rip)
	movb	$0, 393+ST(%rip)
	movl	$0, 396+ST(%rip)
	movb	$0, 400+ST(%rip)
	movb	$0, 401+ST(%rip)
	movb	$0, 402+ST(%rip)
	movb	$0, 403+ST(%rip)
	movb	$0, 404+ST(%rip)
	movb	$0, 405+ST(%rip)
	movb	$0, 406+ST(%rip)
	movb	$0, 407+ST(%rip)
	movb	$0, 408+ST(%rip)
	movb	$0, 409+ST(%rip)
	movl	$0, 412+ST(%rip)
	movb	$0, 416+ST(%rip)
	movb	$0, 417+ST(%rip)
	movb	$0, 418+ST(%rip)
	movb	$0, 419+ST(%rip)
	movb	$0, 420+ST(%rip)
	movb	$0, 421+ST(%rip)
	movb	$0, 422+ST(%rip)
	movb	$0, 423+ST(%rip)
	movb	$0, 424+ST(%rip)
	movb	$0, 425+ST(%rip)
	movl	$0, 428+ST(%rip)
	movb	$0, 432+ST(%rip)
	movb	$0, 433+ST(%rip)
	movb	$0, 434+ST(%rip)
	movb	$0, 435+ST(%rip)
	movb	$0, 436+ST(%rip)
	movb	$0, 437+ST(%rip)
	movb	$0, 438+ST(%rip)
	movb	$0, 439+ST(%rip)
	movb	$0, 440+ST(%rip)
	movb	$0, 441+ST(%rip)
	movl	$0, 444+ST(%rip)
	movb	$0, 448+ST(%rip)
	movb	$0, 449+ST(%rip)
	movb	$0, 450+ST(%rip)
	movb	$0, 451+ST(%rip)
	movb	$0, 452+ST(%rip)
	movb	$0, 453+ST(%rip)
	movb	$0, 454+ST(%rip)
	movb	$0, 455+ST(%rip)
	movb	$0, 456+ST(%rip)
	movb	$0, 457+ST(%rip)
	movl	$0, 460+ST(%rip)
	movb	$0, 464+ST(%rip)
	movb	$0, 465+ST(%rip)
	movb	$0, 466+ST(%rip)
	movb	$0, 467+ST(%rip)
	movb	$0, 468+ST(%rip)
	movb	$0, 469+ST(%rip)
	movb	$0, 470+ST(%rip)
	movb	$0, 471+ST(%rip)
	movb	$0, 472+ST(%rip)
	movb	$0, 473+ST(%rip)
	movl	$0, 476+ST(%rip)
	nop
.L68:
	movl	$0, opd(%rip)
	nop
.L69:
	movl	$0, size(%rip)
	nop
.L70:
	movl	$0, flag(%rip)
	nop
.L71:
	movl	$0, j(%rip)
	nop
.L72:
	movl	$0, i(%rip)
	nop
.L73:
	movl	$-1, o(%rip)
	nop
.L74:
	movl	$-1, s(%rip)
	nop
.L75:
	movl	$0, len(%rip)
	nop
.L76:
	movl	$0, start(%rip)
	nop
.L77:
	movl	$0, locctr(%rip)
	nop
.L78:
	movl	$0, -84(%rbp)
	jmp	.L79
.L80:
	movl	-84(%rbp), %eax
	cltq
	leaq	t3(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -84(%rbp)
.L79:
	cmpl	$9, -84(%rbp)
	jle	.L80
	nop
.L81:
	movl	$0, -80(%rbp)
	jmp	.L82
.L83:
	movl	-80(%rbp), %eax
	cltq
	leaq	t2(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -80(%rbp)
.L82:
	cmpl	$19, -80(%rbp)
	jle	.L83
	nop
.L84:
	movl	$0, -76(%rbp)
	jmp	.L85
.L86:
	movl	-76(%rbp), %eax
	cltq
	leaq	t1(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -76(%rbp)
.L85:
	cmpl	$19, -76(%rbp)
	jle	.L86
	nop
.L87:
	movl	$0, -72(%rbp)
	jmp	.L88
.L89:
	movl	-72(%rbp), %eax
	cltq
	leaq	label(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -72(%rbp)
.L88:
	cmpl	$9, -72(%rbp)
	jle	.L89
	nop
.L90:
	movl	$0, -68(%rbp)
	jmp	.L91
.L92:
	movl	-68(%rbp), %eax
	cltq
	leaq	operand(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -68(%rbp)
.L91:
	cmpl	$9, -68(%rbp)
	jle	.L92
	nop
.L93:
	movl	$0, -64(%rbp)
	jmp	.L94
.L95:
	movl	-64(%rbp), %eax
	cltq
	leaq	opcode(%rip), %rdx
	movb	$0, (%rax,%rdx)
	addl	$1, -64(%rbp)
.L94:
	cmpl	$9, -64(%rbp)
	jle	.L95
	nop
.L96:
	movq	$0, fp5(%rip)
	nop
.L97:
	movq	$0, fp4(%rip)
	nop
.L98:
	movq	$0, fp3(%rip)
	nop
.L99:
	movq	$0, fp2(%rip)
	nop
.L100:
	movq	$0, fp1(%rip)
	nop
.L101:
	movq	$0, _TIG_IZ_NSyn_envp(%rip)
	nop
.L102:
	movq	$0, _TIG_IZ_NSyn_argv(%rip)
	nop
.L103:
	movl	$0, _TIG_IZ_NSyn_argc(%rip)
	nop
	nop
.L104:
.L105:
#APP
# 1308 "jebinshaju_system_software_Pass2.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-NSyn--0
# 0 "" 2
#NO_APP
	movl	-100(%rbp), %eax
	movl	%eax, _TIG_IZ_NSyn_argc(%rip)
	movq	-112(%rbp), %rax
	movq	%rax, _TIG_IZ_NSyn_argv(%rip)
	movq	-120(%rbp), %rax
	movq	%rax, _TIG_IZ_NSyn_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L187:
	cmpq	$68, -16(%rbp)
	ja	.L189
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L108(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L108(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L108:
	.long	.L154-.L108
	.long	.L153-.L108
	.long	.L152-.L108
	.long	.L151-.L108
	.long	.L189-.L108
	.long	.L150-.L108
	.long	.L149-.L108
	.long	.L148-.L108
	.long	.L189-.L108
	.long	.L189-.L108
	.long	.L147-.L108
	.long	.L146-.L108
	.long	.L145-.L108
	.long	.L144-.L108
	.long	.L189-.L108
	.long	.L143-.L108
	.long	.L142-.L108
	.long	.L141-.L108
	.long	.L140-.L108
	.long	.L139-.L108
	.long	.L138-.L108
	.long	.L137-.L108
	.long	.L189-.L108
	.long	.L136-.L108
	.long	.L189-.L108
	.long	.L135-.L108
	.long	.L189-.L108
	.long	.L189-.L108
	.long	.L134-.L108
	.long	.L133-.L108
	.long	.L132-.L108
	.long	.L131-.L108
	.long	.L130-.L108
	.long	.L189-.L108
	.long	.L189-.L108
	.long	.L129-.L108
	.long	.L189-.L108
	.long	.L128-.L108
	.long	.L127-.L108
	.long	.L189-.L108
	.long	.L126-.L108
	.long	.L125-.L108
	.long	.L189-.L108
	.long	.L189-.L108
	.long	.L124-.L108
	.long	.L189-.L108
	.long	.L123-.L108
	.long	.L122-.L108
	.long	.L189-.L108
	.long	.L189-.L108
	.long	.L121-.L108
	.long	.L120-.L108
	.long	.L119-.L108
	.long	.L189-.L108
	.long	.L118-.L108
	.long	.L117-.L108
	.long	.L116-.L108
	.long	.L115-.L108
	.long	.L189-.L108
	.long	.L114-.L108
	.long	.L113-.L108
	.long	.L189-.L108
	.long	.L189-.L108
	.long	.L112-.L108
	.long	.L111-.L108
	.long	.L189-.L108
	.long	.L110-.L108
	.long	.L109-.L108
	.long	.L107-.L108
	.text
.L140:
	movl	s(%rip), %eax
	addl	$1, %eax
	movl	%eax, s(%rip)
	movl	s(%rip), %eax
	cltq
	salq	$4, %rax
	movq	%rax, %rdx
	leaq	ST(%rip), %rax
	addq	%rdx, %rax
	leaq	label(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcpy@PLT
	movl	s(%rip), %edx
	movl	locctr(%rip), %eax
	movslq	%edx, %rdx
	movq	%rdx, %rcx
	salq	$4, %rcx
	leaq	12+ST(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movq	$11, -16(%rbp)
	jmp	.L155
.L121:
	leaq	.LC11(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -32(%rbp)
	movq	$66, -16(%rbp)
	jmp	.L155
.L135:
	leaq	operand(%rip), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -20(%rbp)
	movl	locctr(%rip), %edx
	movl	-20(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, locctr(%rip)
	movq	$64, -16(%rbp)
	jmp	.L155
.L119:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -44(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L155
.L132:
	movzbl	operand(%rip), %eax
	cmpb	$99, %al
	jne	.L156
	movq	$15, -16(%rbp)
	jmp	.L155
.L156:
	movq	$20, -16(%rbp)
	jmp	.L155
.L143:
	movl	len(%rip), %eax
	subl	$3, %eax
	movl	%eax, len(%rip)
	movq	$37, -16(%rbp)
	jmp	.L155
.L116:
	movl	locctr(%rip), %eax
	addl	$3, %eax
	movl	%eax, locctr(%rip)
	movl	size(%rip), %eax
	addl	$3, %eax
	movl	%eax, size(%rip)
	movl	$1, flag(%rip)
	movq	$17, -16(%rbp)
	jmp	.L155
.L131:
	cmpl	$0, -56(%rbp)
	jne	.L158
	movq	$13, -16(%rbp)
	jmp	.L155
.L158:
	movq	$28, -16(%rbp)
	jmp	.L155
.L145:
	cmpl	$0, -36(%rbp)
	jne	.L160
	movq	$51, -16(%rbp)
	jmp	.L155
.L160:
	movq	$0, -16(%rbp)
	jmp	.L155
.L118:
	cmpl	$0, -48(%rbp)
	jne	.L162
	movq	$56, -16(%rbp)
	jmp	.L155
.L162:
	movq	$7, -16(%rbp)
	jmp	.L155
.L153:
	movl	size(%rip), %edx
	movl	locctr(%rip), %eax
	movl	start(%rip), %ecx
	subl	%ecx, %eax
	movl	%eax, %esi
	movq	fp5(%rip), %rax
	movl	%edx, %ecx
	movl	%esi, %edx
	leaq	.LC19(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	fp1(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	fp2(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	fp3(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	fp4(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	fp5(%rip), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	call	pass2
	movq	$41, -16(%rbp)
	jmp	.L155
.L136:
	movl	i(%rip), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	leaq	OT(%rip), %rdx
	addq	%rdx, %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -48(%rbp)
	movq	$54, -16(%rbp)
	jmp	.L155
.L151:
	movzbl	operand(%rip), %eax
	cmpb	$67, %al
	jne	.L164
	movq	$16, -16(%rbp)
	jmp	.L155
.L164:
	movq	$30, -16(%rbp)
	jmp	.L155
.L142:
	movl	len(%rip), %eax
	subl	$3, %eax
	movl	%eax, len(%rip)
	movq	$37, -16(%rbp)
	jmp	.L155
.L137:
	cmpl	$0, -44(%rbp)
	jne	.L166
	movq	$2, -16(%rbp)
	jmp	.L155
.L166:
	movq	$64, -16(%rbp)
	jmp	.L155
.L115:
	movl	locctr(%rip), %edx
	movq	fp4(%rip), %rax
	leaq	operand(%rip), %r9
	leaq	opcode(%rip), %r8
	leaq	label(%rip), %rcx
	leaq	.LC20(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	leaq	.LC21(%rip), %rax
	movq	%rax, %rsi
	leaq	label(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -52(%rbp)
	movq	$38, -16(%rbp)
	jmp	.L155
.L107:
	movl	$0, i(%rip)
	movq	$60, -16(%rbp)
	jmp	.L155
.L146:
	movl	$0, flag(%rip)
	movl	$0, i(%rip)
	movq	$10, -16(%rbp)
	jmp	.L155
.L144:
	leaq	.LC22(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$0, %edi
	call	exit@PLT
.L112:
	leaq	.LC23(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -36(%rbp)
	movq	$12, -16(%rbp)
	jmp	.L155
.L120:
	leaq	operand(%rip), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %edx
	movl	%edx, %eax
	addl	%eax, %eax
	addl	%eax, %edx
	movl	locctr(%rip), %eax
	addl	%edx, %eax
	movl	%eax, locctr(%rip)
	movq	$64, -16(%rbp)
	jmp	.L155
.L139:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -28(%rbp)
	movq	$6, -16(%rbp)
	jmp	.L155
.L130:
	movl	i(%rip), %edx
	movl	s(%rip), %eax
	cmpl	%eax, %edx
	jg	.L168
	movq	$29, -16(%rbp)
	jmp	.L155
.L168:
	movq	$1, -16(%rbp)
	jmp	.L155
.L141:
	movl	flag(%rip), %eax
	testl	%eax, %eax
	jne	.L170
	movq	$50, -16(%rbp)
	jmp	.L155
.L170:
	movq	$64, -16(%rbp)
	jmp	.L155
.L126:
	movl	locctr(%rip), %eax
	addl	$3, %eax
	movl	%eax, locctr(%rip)
	movl	size(%rip), %eax
	addl	$3, %eax
	movl	%eax, size(%rip)
	movq	$64, -16(%rbp)
	jmp	.L155
.L109:
	movl	opd(%rip), %eax
	movl	%eax, start(%rip)
	movl	start(%rip), %eax
	movl	%eax, locctr(%rip)
	movl	opd(%rip), %edx
	movq	fp4(%rip), %rax
	movl	%edx, %r8d
	leaq	opcode(%rip), %rdx
	movq	%rdx, %rcx
	leaq	label(%rip), %rdx
	leaq	.LC24(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	fp1(%rip), %rax
	leaq	operand(%rip), %r8
	leaq	opcode(%rip), %rdx
	movq	%rdx, %rcx
	leaq	label(%rip), %rdx
	leaq	.LC25(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movq	$19, -16(%rbp)
	jmp	.L155
.L117:
	movq	fp4(%rip), %rax
	leaq	operand(%rip), %r8
	leaq	opcode(%rip), %rdx
	movq	%rdx, %rcx
	leaq	label(%rip), %rdx
	leaq	.LC26(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$0, i(%rip)
	movq	$32, -16(%rbp)
	jmp	.L155
.L113:
	movl	i(%rip), %edx
	movl	s(%rip), %eax
	cmpl	%eax, %edx
	jg	.L172
	movq	$46, -16(%rbp)
	jmp	.L155
.L172:
	movq	$18, -16(%rbp)
	jmp	.L155
.L114:
	cmpl	$0, -60(%rbp)
	jne	.L174
	movq	$67, -16(%rbp)
	jmp	.L155
.L174:
	movq	$44, -16(%rbp)
	jmp	.L155
.L149:
	cmpl	$0, -28(%rbp)
	je	.L176
	movq	$57, -16(%rbp)
	jmp	.L155
.L176:
	movq	$55, -16(%rbp)
	jmp	.L155
.L127:
	cmpl	$0, -52(%rbp)
	je	.L178
	movq	$68, -16(%rbp)
	jmp	.L155
.L178:
	movq	$11, -16(%rbp)
	jmp	.L155
.L134:
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$60, -16(%rbp)
	jmp	.L155
.L122:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp1(%rip)
	leaq	.LC1(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp2(%rip)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp3(%rip)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp4(%rip)
	leaq	.LC4(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC29(%rip), %rax
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, fp5(%rip)
	call	read_OPTAB
	movq	fp1(%rip), %rax
	leaq	opd(%rip), %r8
	leaq	opcode(%rip), %rdx
	movq	%rdx, %rcx
	leaq	label(%rip), %rdx
	leaq	.LC30(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	leaq	.LC31(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -60(%rbp)
	movq	$59, -16(%rbp)
	jmp	.L155
.L124:
	movl	$0, locctr(%rip)
	movq	$19, -16(%rbp)
	jmp	.L155
.L150:
	movq	$47, -16(%rbp)
	jmp	.L155
.L128:
	movl	locctr(%rip), %edx
	movl	len(%rip), %eax
	addl	%edx, %eax
	movl	%eax, locctr(%rip)
	movl	size(%rip), %edx
	movl	len(%rip), %eax
	addl	%edx, %eax
	movl	%eax, size(%rip)
	movq	$64, -16(%rbp)
	jmp	.L155
.L111:
	movq	fp1(%rip), %rax
	leaq	operand(%rip), %r8
	leaq	opcode(%rip), %rdx
	movq	%rdx, %rcx
	leaq	label(%rip), %rdx
	leaq	.LC25(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movq	$19, -16(%rbp)
	jmp	.L155
.L125:
	movl	$0, %eax
	jmp	.L188
.L147:
	movl	i(%rip), %edx
	movl	o(%rip), %eax
	cmpl	%eax, %edx
	jg	.L181
	movq	$23, -16(%rbp)
	jmp	.L155
.L181:
	movq	$17, -16(%rbp)
	jmp	.L155
.L154:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rsi
	leaq	opcode(%rip), %rax
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -40(%rbp)
	movq	$35, -16(%rbp)
	jmp	.L155
.L123:
	movl	i(%rip), %eax
	cltq
	salq	$4, %rax
	movq	%rax, %rdx
	leaq	ST(%rip), %rax
	addq	%rdx, %rax
	leaq	label(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -56(%rbp)
	movq	$31, -16(%rbp)
	jmp	.L155
.L110:
	cmpl	$0, -32(%rbp)
	jne	.L183
	movq	$40, -16(%rbp)
	jmp	.L155
.L183:
	movq	$63, -16(%rbp)
	jmp	.L155
.L148:
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$10, -16(%rbp)
	jmp	.L155
.L129:
	cmpl	$0, -40(%rbp)
	jne	.L185
	movq	$25, -16(%rbp)
	jmp	.L155
.L185:
	movq	$52, -16(%rbp)
	jmp	.L155
.L133:
	movl	i(%rip), %eax
	cltq
	salq	$4, %rax
	movq	%rax, %rdx
	leaq	12+ST(%rip), %rax
	movl	(%rdx,%rax), %edx
	movl	i(%rip), %eax
	cltq
	salq	$4, %rax
	movq	%rax, %rcx
	leaq	ST(%rip), %rax
	leaq	(%rcx,%rax), %rsi
	movq	fp3(%rip), %rax
	movl	%edx, %ecx
	movq	%rsi, %rdx
	leaq	.LC33(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	i(%rip), %eax
	addl	$1, %eax
	movl	%eax, i(%rip)
	movq	$32, -16(%rbp)
	jmp	.L155
.L152:
	leaq	operand(%rip), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movl	%eax, len(%rip)
	movq	$3, -16(%rbp)
	jmp	.L155
.L138:
	movl	len(%rip), %eax
	subl	$3, %eax
	movl	%eax, %edx
	shrl	$31, %edx
	addl	%edx, %eax
	sarl	%eax
	movl	%eax, len(%rip)
	movq	$37, -16(%rbp)
	jmp	.L155
.L189:
	nop
.L155:
	jmp	.L187
.L188:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
	.globl	read_OPTAB
	.type	read_OPTAB, @function
read_OPTAB:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$3, -8(%rbp)
.L200:
	cmpq	$4, -8(%rbp)
	je	.L191
	cmpq	$4, -8(%rbp)
	ja	.L201
	cmpq	$3, -8(%rbp)
	je	.L193
	cmpq	$3, -8(%rbp)
	ja	.L201
	cmpq	$0, -8(%rbp)
	je	.L202
	cmpq	$2, -8(%rbp)
	je	.L195
	jmp	.L201
.L191:
	cmpl	$-1, -12(%rbp)
	jne	.L196
	movq	$0, -8(%rbp)
	jmp	.L198
.L196:
	movq	$2, -8(%rbp)
	jmp	.L198
.L193:
	movq	$2, -8(%rbp)
	jmp	.L198
.L195:
	movl	o(%rip), %eax
	addl	$1, %eax
	movl	%eax, o(%rip)
	movl	o(%rip), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	leaq	OT(%rip), %rdx
	addq	%rdx, %rax
	leaq	10(%rax), %rcx
	movl	o(%rip), %eax
	movslq	%eax, %rdx
	movq	%rdx, %rax
	salq	$2, %rax
	addq	%rdx, %rax
	salq	$2, %rax
	leaq	OT(%rip), %rdx
	addq	%rax, %rdx
	movq	fp2(%rip), %rax
	leaq	.LC12(%rip), %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_fscanf@PLT
	movq	fp2(%rip), %rax
	movq	%rax, %rdi
	call	getc@PLT
	movl	%eax, -12(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L198
.L201:
	nop
.L198:
	jmp	.L200
.L202:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	read_OPTAB, .-read_OPTAB
	.globl	search_SYMTAB
	.type	search_SYMTAB, @function
search_SYMTAB:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$6, -8(%rbp)
.L219:
	cmpq	$8, -8(%rbp)
	ja	.L220
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L206(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L206(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L206:
	.long	.L212-.L206
	.long	.L211-.L206
	.long	.L220-.L206
	.long	.L210-.L206
	.long	.L209-.L206
	.long	.L208-.L206
	.long	.L207-.L206
	.long	.L220-.L206
	.long	.L205-.L206
	.text
.L209:
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	movq	%rax, %rdx
	leaq	ST(%rip), %rax
	addq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	strcmp@PLT
	movl	%eax, -12(%rbp)
	movq	$3, -8(%rbp)
	jmp	.L213
.L205:
	movl	s(%rip), %eax
	cmpl	%eax, -16(%rbp)
	jg	.L214
	movq	$4, -8(%rbp)
	jmp	.L213
.L214:
	movq	$5, -8(%rbp)
	jmp	.L213
.L211:
	addl	$1, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L213
.L210:
	cmpl	$0, -12(%rbp)
	jne	.L216
	movq	$0, -8(%rbp)
	jmp	.L213
.L216:
	movq	$1, -8(%rbp)
	jmp	.L213
.L207:
	movl	$0, -16(%rbp)
	movq	$8, -8(%rbp)
	jmp	.L213
.L208:
	movl	$-1, %eax
	jmp	.L218
.L212:
	movl	-16(%rbp), %eax
	cltq
	salq	$4, %rax
	movq	%rax, %rdx
	leaq	12+ST(%rip), %rax
	movl	(%rdx,%rax), %eax
	jmp	.L218
.L220:
	nop
.L213:
	jmp	.L219
.L218:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	search_SYMTAB, .-search_SYMTAB
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
