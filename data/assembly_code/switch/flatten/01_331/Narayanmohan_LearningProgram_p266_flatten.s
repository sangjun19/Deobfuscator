	.file	"Narayanmohan_LearningProgram_p266_flatten.c"
	.text
	.globl	_TIG_IZ_niQF_envp
	.bss
	.align 8
	.type	_TIG_IZ_niQF_envp, @object
	.size	_TIG_IZ_niQF_envp, 8
_TIG_IZ_niQF_envp:
	.zero	8
	.globl	_TIG_IZ_niQF_argc
	.align 4
	.type	_TIG_IZ_niQF_argc, @object
	.size	_TIG_IZ_niQF_argc, 4
_TIG_IZ_niQF_argc:
	.zero	4
	.globl	_TIG_IZ_niQF_argv
	.align 8
	.type	_TIG_IZ_niQF_argv, @object
	.size	_TIG_IZ_niQF_argv, 8
_TIG_IZ_niQF_argv:
	.zero	8
	.section	.rodata
	.align 8
.LC0:
	.string	"\n Not a valid Single Digit Number"
.LC1:
	.string	"Six"
.LC2:
	.string	"Three"
.LC3:
	.string	"ONE"
.LC4:
	.string	"Eight"
.LC5:
	.string	"zero"
.LC6:
	.string	"Two"
.LC7:
	.string	"Nine"
.LC8:
	.string	"Four"
.LC9:
	.string	"Seven"
.LC10:
	.string	"Five"
	.text
	.globl	printinword
	.type	printinword, @function
printinword:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$10, -8(%rbp)
.L32:
	cmpq	$22, -8(%rbp)
	ja	.L33
	movq	-8(%rbp), %rax
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
	.long	.L16-.L4
	.long	.L15-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L33-.L4
	.long	.L33-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L33-.L4
	.long	.L33-.L4
	.long	.L10-.L4
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L33-.L4
	.long	.L7-.L4
	.long	.L33-.L4
	.long	.L33-.L4
	.long	.L6-.L4
	.long	.L33-.L4
	.long	.L33-.L4
	.long	.L33-.L4
	.long	.L5-.L4
	.long	.L34-.L4
	.text
.L7:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L8:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L15:
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L13:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L5:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L9:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L6:
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L12:
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L10:
	cmpl	$9, -20(%rbp)
	ja	.L19
	movl	-20(%rbp), %eax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L20-.L21
	.text
.L20:
	movq	$6, -8(%rbp)
	jmp	.L31
.L22:
	movq	$21, -8(%rbp)
	jmp	.L31
.L23:
	movq	$7, -8(%rbp)
	jmp	.L31
.L24:
	movq	$12, -8(%rbp)
	jmp	.L31
.L25:
	movq	$2, -8(%rbp)
	jmp	.L31
.L26:
	movq	$0, -8(%rbp)
	jmp	.L31
.L27:
	movq	$1, -8(%rbp)
	jmp	.L31
.L28:
	movq	$17, -8(%rbp)
	jmp	.L31
.L29:
	movq	$3, -8(%rbp)
	jmp	.L31
.L30:
	movq	$11, -8(%rbp)
	jmp	.L31
.L19:
	movq	$14, -8(%rbp)
	nop
.L31:
	jmp	.L17
.L16:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L11:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L14:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$22, -8(%rbp)
	jmp	.L17
.L33:
	nop
.L17:
	jmp	.L32
.L34:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	printinword, .-printinword
	.section	.rodata
.LC11:
	.string	"%d"
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
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, _TIG_IZ_niQF_envp(%rip)
	nop
.L36:
	movq	$0, _TIG_IZ_niQF_argv(%rip)
	nop
.L37:
	movl	$0, _TIG_IZ_niQF_argc(%rip)
	nop
	nop
.L38:
.L39:
#APP
# 103 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-niQF--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_niQF_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_niQF_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_niQF_envp(%rip)
	nop
	movq	$2, -16(%rbp)
.L45:
	cmpq	$2, -16(%rbp)
	je	.L40
	cmpq	$2, -16(%rbp)
	ja	.L48
	cmpq	$0, -16(%rbp)
	je	.L42
	cmpq	$1, -16(%rbp)
	jne	.L48
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-20(%rbp), %eax
	movl	%eax, %edi
	call	printinword
	movq	$0, -16(%rbp)
	jmp	.L43
.L42:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L46
	jmp	.L47
.L40:
	movq	$1, -16(%rbp)
	jmp	.L43
.L48:
	nop
.L43:
	jmp	.L45
.L47:
	call	__stack_chk_fail@PLT
.L46:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	main, .-main
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
